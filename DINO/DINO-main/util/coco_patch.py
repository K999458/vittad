import numpy as np
from pycocotools.cocoeval import COCOeval

# 保存原始的accumulate方法
original_accumulate = COCOeval.accumulate

def accumulate_patch(self):
    """
    修补后的accumulate方法，将np.float替换为np.float64
    """
    def patch_code(code):
        if isinstance(code, np.ndarray):
            return code.astype(np.float64)
        return code

    # 调用原始方法
    try:
        return original_accumulate(self)
    except AttributeError as e:
        if "float" in str(e):
            # 修补关键数组的类型
            if hasattr(self, 'eval'):
                self.eval = patch_code(self.eval)
            if hasattr(self, '_gts'):
                self._gts = patch_code(self._gts)
            if hasattr(self, '_dts'):
                self._dts = patch_code(self._dts)
            return original_accumulate(self)
        raise

# 应用补丁
COCOeval.accumulate = accumulate_patch 