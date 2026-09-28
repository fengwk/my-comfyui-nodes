import { app } from "../../scripts/app.js";

// Draw in processing order without reordering node.widgets: workflows restore
// their positional values from that array.
const order = [
    "_ve_dlss",
    "enable_super_resolution", "spatial_mode", "sr_preset",
    "enable_neural_rendering", "nr_profile", "nr_intensity",
    "detail", "color", "_ve_custom", "style", "preset", "local_structure",
    "local_tone", "skin", "ui_correction", "auto_mask",
    "_ve_common", "motion", "scene_cut_threshold", "channel_order",
    "_ve_runtime", "gpu_index", "runtime_dir", "wine_prefix", "worker_timeout",
    "_ve_gimm",
    "enable_frame_interpolation", "vfi_precision", "vfi_ds_factor",
    "_ve_pipeline", "stage_order",
    "_ve_output", "output_codec", "quality", "_ve_passthrough",
];
const positions = new Map(order.map((name, index) => [name, index]));

const labels = {
    enable_super_resolution: "超分 / DLAA",
    spatial_mode: "  放大倍率",
    sr_preset: "  SR 模型",
    enable_neural_rendering: "神经渲染 NR",
    nr_profile: "  配置",
    nr_intensity: "  模型强度",
    detail: "  结果混合强度",
    color: "  NR 颜色比例",
    style: "  风格",
    preset: "  模型档位",
    local_structure: "  局部结构",
    local_tone: "  局部影调",
    skin: "  皮肤结构",
    ui_correction: "  UI 校正",
    auto_mask: "  自动遮罩",
    enable_frame_interpolation: "启用插帧",
    vfi_precision: "  精度",
    vfi_ds_factor: "  光流缩放",
    stage_order: "阶段顺序",
    motion: "  运动引导",
    scene_cut_threshold: "  切镜阈值",
    channel_order: "  通道顺序",
    output_codec: "编码器",
    quality: "质量（数值越低质量越高）",
    gpu_index: "  DLSS GPU",
    runtime_dir: "  NVIDIA DLL 目录",
    wine_prefix: "  Wine 前缀",
    worker_timeout: "  单次等待超时（秒）",
};

// Passive headings are appended, excluded from persistence and API prompts.
const sections = [
    ["dlss", "DLSS 增强"],
    ["custom", "NR 自定义模型"],
    ["common", "DLSS 共用设置"],
    ["runtime", "DLSS 运行环境"],
    ["gimm", "GIMM 插帧"],
    ["pipeline", "处理顺序"],
    ["output", "输出编码"],
    ["passthrough", "原样输出 · 不执行增强"],
];

const customFields = new Set([
    "style", "preset", "local_structure", "local_tone", "skin",
    "ui_correction", "auto_mask",
]);
const srFields = new Set(["spatial_mode", "sr_preset"]);
const nrFields = new Set(["nr_profile", "nr_intensity", "detail", "color"]);
const vfiFields = new Set(["vfi_precision", "vfi_ds_factor"]);
const dlssFields = new Set([
    "motion", "channel_order", "runtime_dir", "wine_prefix",
    "worker_timeout", "gpu_index",
]);

function active(name, values) {
    const dlss = values.enable_super_resolution || values.enable_neural_rendering;
    if (srFields.has(name)) return !!values.enable_super_resolution;
    if (nrFields.has(name)) return !!values.enable_neural_rendering;
    if (customFields.has(name)) return !!values.enable_neural_rendering && values.nr_profile === "custom";
    if (vfiFields.has(name)) return !!values.enable_frame_interpolation;
    if (dlssFields.has(name)) return !!dlss;
    if (name === "scene_cut_threshold") return !!dlss && values.motion === "optical_flow";
    if (name === "stage_order") return !!dlss && !!values.enable_frame_interpolation;
    if (["output_codec", "quality"].includes(name)) return !!dlss || !!values.enable_frame_interpolation;
    return true;
}

function update(node) {
    if (!node.widgets || !node._videoEnhanceSections) return;
    const values = Object.fromEntries(node.widgets.map((widget) => [widget.name, widget.value]));
    const linked = new Set((node.inputs || [])
        .filter((input) => input.link != null)
        .map((input) => input.widget?.name || input.name));
    // Linked controls are evaluated upstream at execution time, not from the
    // local widget cache. Show all settings that could apply.
    for (const name of ["enable_super_resolution", "enable_neural_rendering",
        "enable_frame_interpolation"]) {
        if (linked.has(name)) values[name] = true;
    }
    if (linked.has("nr_profile")) values.nr_profile = "custom";
    if (linked.has("motion")) values.motion = "optical_flow";
    let changed = false;
    const dlss = values.enable_super_resolution || values.enable_neural_rendering;
    const enabled = dlss || values.enable_frame_interpolation;
    for (const [id] of sections) {
        const section = node._videoEnhanceSections?.[id];
        if (!section) continue;
        const present = id === "custom" ? values.enable_neural_rendering && values.nr_profile === "custom"
            : ["common", "runtime"].includes(id) ? dlss
            : id === "pipeline" ? dlss && values.enable_frame_interpolation
            : id === "output" ? enabled && values.output_codec !== undefined
            : id === "passthrough" ? !enabled : true;
        const hidden = !present;
        if (Boolean(section.widget.hidden) !== Boolean(hidden)) changed = true;
        section.widget.hidden = Boolean(hidden);
    }
    for (const widget of node.widgets) {
        const original = widget._videoEnhanceOriginal;
        if (!original) continue;
        // Native hidden is presentation-only. Never impersonate a converted
        // input or change type, size callbacks, or either serialization flag.
        const hidden = original.hidden || !(linked.has(widget.name) || active(widget.name, values));
        if (Boolean(widget.hidden) === Boolean(hidden)) continue;
        widget.hidden = hidden;
        changed = true;
    }
    // Preserve the node's width while recomputing its height for the visible fields.
    if (changed && node.computeSize && node.setSize) {
        const computed = node.computeSize();
        node.setSize([Math.max(node.size?.[0] || 0, computed[0]), computed[1]]);
    }
    node.setDirtyCanvas?.(true, true);
}

function attach(node) {
    if (!node.widgets || node._videoEnhanceLayoutAttached) return;
    node._videoEnhanceLayoutAttached = true;
    node._videoEnhanceSections = {};
    for (const [id, title] of sections) {
        const widget = node.addCustomWidget({
            name: `_ve_${id}`,
            type: "video_enhance_heading",
            label: title,
            serialize: false,
            options: { serialize: false },
            computeSize: () => [0, 26],
            draw(ctx, _node, width, y) {
                ctx.save();
                ctx.fillStyle = "#b8c4d0";
                ctx.font = "bold 12px sans-serif";
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText(title, 14, y + 15);
                const end = 24 + ctx.measureText(title).width;
                if (end < width - 14) {
                    ctx.strokeStyle = "#555";
                    ctx.beginPath();
                    ctx.moveTo(end, y + 15);
                    ctx.lineTo(width - 14, y + 15);
                    ctx.stroke();
                }
                ctx.restore();
            },
        });
        node._videoEnhanceSections[id] = { widget };
    }
    for (const widget of node.widgets) {
        if (!(widget.name in labels)) continue;
        widget.label = labels[widget.name];
        // These sections replace the global advanced toggle for managed fields.
        widget.advanced = false;
        widget._videoEnhanceOriginal = {
            hidden: Boolean(widget.hidden),
        };
        if (["enable_super_resolution", "enable_neural_rendering",
             "enable_frame_interpolation", "nr_profile", "motion"].includes(widget.name)) {
            const callback = widget.callback;
            widget.callback = function (...args) {
                const result = callback?.apply(this, args);
                update(node);
                return result;
            };
        }
    }
    update(node);
}

app.registerExtension({
    name: "my-comfyui-nodes.video-enhance-layout",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["MyVideoEnhance", "MyVideoEnhanceStream"].includes(nodeData.name)) return;
        const layoutWidgets = nodeType.prototype.getLayoutWidgets;
        if (layoutWidgets) {
            nodeType.prototype.getLayoutWidgets = function (...args) {
                const widgets = layoutWidgets.apply(this, args);
                return [...widgets].sort((a, b) =>
                    (positions.get(a.name) ?? order.length) - (positions.get(b.name) ?? order.length));
            };
        }
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            attach(this);
            return result;
        };
        const configured = nodeType.prototype.onGraphConfigured;
        nodeType.prototype.onGraphConfigured = function (...args) {
            const result = configured?.apply(this, args);
            attach(this);
            update(this);
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            attach(this);
            update(this);
            return result;
        };
        const connectionsChanged = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const result = connectionsChanged?.apply(this, args);
            update(this);
            return result;
        };
    },
});
