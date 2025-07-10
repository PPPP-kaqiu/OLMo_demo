from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
# 导入 numpy 用于处理字节数据
import numpy as np

__all__ = ["DataCollator_Stream"]


@dataclass
class DataCollator_Stream:
    @classmethod
    def from_train_config(cls) -> DataCollator:
        # 由于不需要填充，我们不再需要从配置中获取 pad_token_id 等信息。
        return cls()

    def __call__(self, items: Union[List[Dict[str, Any]], List[torch.Tensor]]) -> Dict[str, Any]:
        assert items, "批处理的样本列表不能为空"

        all_input_ids = []

        for item in items:
            if isinstance(item, dict):
                if "input_ids" in item:
                    input_ids = item["input_ids"]
                elif "tokens" in item:
                    token_bytes = item['tokens']
                    input_ids = torch.from_numpy(np.frombuffer(token_bytes, dtype=np.uint16).copy())
                else:
                    raise KeyError("样本字典中必须包含 'input_ids' 或 'tokens' 键。")
            else:
                input_ids = item

            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            
            all_input_ids.append(input_ids.to(dtype=torch.long))
            
        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}

        return out