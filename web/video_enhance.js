import { app } from "../../scripts/app.js";

// Only presentation changes: keep every widget in its original array position
// and preserve its value so old positional workflows and prompt inputs survive.
const labels = {
    enable_super_resolution: "DLSS · Super Resolution",
    spatial_mode: "DLSS · Scale / DLAA",
    sr_preset: "DLSS · SR model",
    enable_neural_rendering: "DLSS · Neural Rendering (experimental)",
    nr_profile: "NR · Profile",
    nr_intensity: "NR · Intensity",
    style: "NR custom · Style",
    preset: "NR custom · Preset",
    local_structure: "NR custom · Structure",
    local_tone: "NR custom · Tone",
    skin: "NR custom · Skin",
    ui_correction: "NR custom · UI correction",
    auto_mask: "NR custom · Auto mask",
    detail: "NR · Detail",
    color: "NR · Color",
    enable_frame_interpolation: "GIMM · Interpolation",
    vfi_precision: "GIMM · Precision",
    vfi_ds_factor: "GIMM · Flow scale",
    stage_order: "Pipeline · Stage order",
    motion: "DLSS · Motion guide",
    scene_cut_threshold: "DLSS · Scene cut",
    channel_order: "DLSS · Channel order",
    runtime_dir: "Runtime · NVIDIA DLL directory",
    wine_prefix: "Runtime · Wine prefix",
    worker_timeout: "Runtime · Worker timeout",
    gpu_index: "Runtime · GPU index",
    output_codec: "Output · Codec",
    quality: "Output · Quality",
};

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
    return true;
}

function update(node) {
    if (!node.widgets) return;
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
    for (const widget of node.widgets) {
        if (!(widget.name in labels)) continue;
        widget.label = labels[widget.name];
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
