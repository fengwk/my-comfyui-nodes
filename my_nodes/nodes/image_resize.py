"""ComfyUI IMAGE long-edge resize node."""

from my_nodes.core.image_resize import resize_long_edge


class MyImageResizeLongEdge:
    NODE_ID = "MyImageResizeLongEdge"
    DISPLAY_NAME = "My Image Resize Long Edge (长边缩放)"
    CATEGORY = "my_nodes/image"
    FUNCTION = "resize"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "long_edge": ("INT", {
                    "default": 640, "min": 1, "max": 16384, "step": 1,
                    "tooltip": "目标长边像素；短边按比例取整，不裁剪、不补边。支持视频帧批次。",
                }),
                "interpolation": (
                    ["bicubic", "bilinear", "nearest", "area"],
                    {"default": "bicubic"},
                ),
            },
            "optional": {
                "only_downscale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "仅缩小：长边不超过目标时原样输出。关闭后允许放大小图。",
                }),
            },
        }

    def resize(self, images, long_edge, interpolation, only_downscale=True):
        from comfy.model_management import throw_exception_if_processing_interrupted

        return (resize_long_edge(
            images, long_edge, interpolation,
            check_interrupt=throw_exception_if_processing_interrupted,
            only_downscale=only_downscale,
        ),)
