"""JSONL 数据与有证据位置的确定性合成检索。"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def read_samples(path, ids=(), limit=None):
    """读取 JSONL 样本，检查 id/prompt 字段，确保唯一性和存在性"""
    samples, seen = [], set()
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        s = json.loads(line)
        if not isinstance(s.get("id"), str) or not isinstance(s.get("prompt"), str):
            raise ValueError(f"JSONL 第 {line_no} 行必须包含字符串 id/prompt")
        if s["id"] in seen:
            raise ValueError(f"重复样本 ID: {s['id']}")
        seen.add(s["id"])
        if not ids or s["id"] in ids:
            samples.append(s)
    missing = set(ids) - seen
    if missing:
        raise ValueError(f"样本 ID 不存在：{sorted(missing)}")
    if limit:
        samples = samples[:limit]
    if not samples:
        raise ValueError("数据筛选后为空")
    return samples


def synthetic(path, count=8, records=128, seed=42):
    """ 生成确定性的检索/三跳关联检索任务，带有证据位置，便于调试和性能分析 """
    if count < 1 or records < 4:
        raise ValueError("count>=1 且 records>=4")
    output = []
    for i in range(count):
        rng = random.Random(seed + i)
        names = [f"entity_{j:06d}" for j in range(records)]
        values = [f"code_{rng.randrange(10**9):09d}" for _ in names]
        task = "retrieval" if i % 2 == 0 else "multihop"
        target = rng.randrange(records)
        facts = [f"{name} has code {val}." for name, val in zip(names, values)]
        evidence_text = [facts[target]]
        if task == "multihop":
            # 三跳：入口 -> alias -> entity -> code；干扰项不重用 alias。
            alias = f"alias_{i:06d}"
            start = f"entry_{i:06d}"
            evidence_text += [f"{start} points to {alias}.", f"{alias} points to {names[target]}."]
            facts.extend(evidence_text[1:])
            question = f"Follow the links from {start}. What is the final entity's code?"
        else:
            question = f"What is the code of {names[target]}?"
        rng.shuffle(facts)
        header = "Read the facts and answer the question. Return only the exact code, no explanation.\nFacts:\n"
        body = "\n".join(facts)
        prompt = header + body + "\nQuestion: " + question + "\nAnswer:"
        evidence = []
        for fact in evidence_text:
            start_char = prompt.index(fact)
            evidence.append({"text": fact, "record_index": facts.index(fact),
                             "char_start": start_char, "char_end": start_char + len(fact)})
        output.append({"id": f"s{i:04d}", "prompt": prompt, "reference": values[target],
                       "task": task, "metadata": {"seed": seed + i, "generator": "facts-v1",
                       "rule": "unique entity->code; multihop entry->alias->entity->code",
                       "records": len(facts), "evidence": evidence}})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in output))
    return output


def prepare(tokenizer, sample, config, target_length=None):
    """分词，精确目标长度仅加前置无关填充；过长明确失败，不截断任务证据。"""
    d = config["data"]

    def encode(prompt):
        if d["chat_template"]:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=d["thinking"],
                chat_template=d["template"],
            )
        return tokenizer.encode(prompt, add_special_tokens=True)

    ids = encode(sample["prompt"])
    base_length = len(ids)
    added = 0
    if target_length is not None:
        if len(ids) > target_length:
            raise ValueError(f"样本 {sample['id']} 模板后 {len(ids)} tokens 超过目标 {target_length}；禁止截断")
        padding = tokenizer.encode(d["padding_text"], add_special_tokens=False)
        if not padding and len(ids) < target_length:
            raise ValueError("padding_text 未产生 token")
        added = target_length - len(ids)
        # 填充放在整个模板前，保留模板、问题和所有证据；完整 token IDs 持久化。
        ids = (padding * ((added + len(padding) - 1) // len(padding)))[:added] + ids if added else ids
    return ids, {"base_tokens": base_length,
                 "prefix_padding_tokens": added,
                 "padding_placement": "before_entire_template",
                 "prompt": sample["prompt"],
                 "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
