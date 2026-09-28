import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { test } from "node:test";

const source = readFileSync(new URL("../web/video_enhance.js", import.meta.url), "utf8")
    .replace(/^import \{ app \} from "\.\.\/\.\.\/scripts\/app\.js";\s*/u, "");
let extension;
runInNewContext(source, {
    app: { registerExtension(value) { extension = value; } },
});

function makeNode(name = "MyVideoEnhanceStream", prepare = () => {}) {
    const nodeType = function () {};
    nodeType.prototype.getLayoutWidgets = function () {
        return this.widgets.filter((widget) => !widget.hidden);
    };
    nodeType.prototype.addCustomWidget = function (widget) {
        this.widgets.push(widget);
        return widget;
    };
    extension.beforeRegisterNodeDef(nodeType, { name });
    const names = [
        "enable_super_resolution", "spatial_mode", "enable_neural_rendering",
        "nr_profile", "nr_intensity", "enable_frame_interpolation",
        "output_codec", "quality", "vfi_precision", "vfi_ds_factor", "motion",
        "scene_cut_threshold", "channel_order", "runtime_dir", "wine_prefix",
        "worker_timeout", "stage_order", "style", "preset", "local_structure",
        "local_tone", "skin", "detail", "color", "ui_correction", "auto_mask",
        "sr_preset", "gpu_index",
    ];
    const values = {
        enable_super_resolution: false, spatial_mode: "2.0x",
        enable_neural_rendering: false, nr_profile: "standard",
        nr_intensity: 1.0, enable_frame_interpolation: false,
        output_codec: "libx264", quality: 18, vfi_precision: "fp32", vfi_ds_factor: 1,
        motion: "optical_flow", scene_cut_threshold: 0.2, channel_order: "auto",
        stage_order: "vfi_then_dlss", style: "Cinematic",
        preset: "Default", local_structure: 1, local_tone: 1, skin: -1,
        detail: 1, color: 1, ui_correction: false, auto_mask: false,
        sr_preset: "Default", gpu_index: 0, runtime_dir: "",
        wine_prefix: "", worker_timeout: 600,
    };
    const node = new nodeType();
    node.widgets = names.map((name) => ({
        name, value: values[name], type: "combo",
        callback() {}, computeSize: () => [120, 24],
    }));
    node.size = [220, 500];
    node.inputs = [];
    node.computeSize = () => [180, 30 + node.widgets.filter(
        (widget) => !widget.hidden,
    ).length * 28];
    node.setSize = (size) => { node.size = size; };
    prepare(node);
    node.onNodeCreated();
    const field = (name) => node.widgets.find((widget) => widget.name === name);
    const change = (name, value) => {
        field(name).value = value;
        field(name).callback(value);
    };
    const visible = (name) => !field(name).hidden;
    return { node, names, field, change, visible };
}

test("only active settings are visible; pass-through hides encoding controls", () => {
    const { change, visible } = makeNode();
    assert.equal(visible("spatial_mode"), false);
    assert.equal(visible("nr_profile"), false);
    assert.equal(visible("vfi_precision"), false);
    assert.equal(visible("stage_order"), false);
    assert.equal(visible("output_codec"), false);
    assert.equal(visible("_ve_passthrough"), true);
    change("enable_neural_rendering", true);
    assert.equal(visible("nr_profile"), true);
    assert.equal(visible("detail"), true);
    assert.equal(visible("style"), false);
    change("nr_profile", "custom");
    assert.equal(visible("style"), true);
    change("motion", "none");
    assert.equal(visible("scene_cut_threshold"), false);
    change("enable_frame_interpolation", true);
    assert.equal(visible("stage_order"), true);
    assert.equal(visible("vfi_precision"), true);
    change("enable_neural_rendering", false);
    assert.equal(visible("stage_order"), false);
    assert.equal(visible("vfi_precision"), true);
});

test("controls are grouped, hidden values survive toggling and restored workflows", () => {
    const { node, names, field, change, visible } = makeNode();
    const initial = node.widgets.map((widget) => widget.serializeValue?.() ?? widget.value);
    change("enable_super_resolution", true);
    change("enable_neural_rendering", true);
    change("enable_frame_interpolation", true);
    const index = (name) => node.getLayoutWidgets().findIndex((widget) => widget.name === name);
    assert.ok(index("detail") < index("enable_frame_interpolation"));
    assert.ok(index("motion") < index("enable_frame_interpolation"));
    assert.ok(index("motion") < index("output_codec"));
    assert.ok(index("wine_prefix") < index("enable_frame_interpolation"));
    assert.ok(index("sr_preset") < index("enable_neural_rendering"));
    change("nr_profile", "custom");
    change("style", "Natural");
    change("nr_profile", "standard");
    assert.equal(visible("style"), false);
    assert.equal(field("style").serializeValue?.() ?? field("style").value, "Natural");
    assert.deepEqual(node.widgets.filter((w) => w.serialize !== false).map((widget) => widget.name), names);
    assert.equal(initial.length, node.widgets.length);
    field("enable_neural_rendering").value = false;
    field("enable_super_resolution").value = true;
    node.onConfigure({});
    assert.equal(visible("spatial_mode"), true);
    assert.equal(visible("nr_profile"), false);
    assert.equal(field("style").value, "Natural");
});

test("NR-only panel automatically shows all active DLSS subgroups before GIMM", () => {
    // Reproduces the screenshot state, including the advanced controls being expanded.
    const { node, change } = makeNode();
    change("enable_neural_rendering", true);
    assert.deepEqual(Array.from(node.getLayoutWidgets(), (w) => w.name), [
        "_ve_dlss",
        "enable_super_resolution",
        "enable_neural_rendering", "nr_profile", "nr_intensity", "detail", "color",
        "_ve_common", "motion", "scene_cut_threshold", "channel_order",
        "_ve_runtime", "gpu_index", "runtime_dir", "wine_prefix", "worker_timeout",
        "_ve_gimm",
        "enable_frame_interpolation",
        "_ve_output",
        "output_codec", "quality",
    ]);
});

test("legacy positional widget values still restore by backend order", () => {
    // Rendering a sorted copy must not change LiteGraph's positional persistence.
    const { node, names } = makeNode("MyVideoEnhanceStream", (node) => {
        node.widgets.find((w) => w.name === "enable_neural_rendering").value = true;
        node.widgets.find((w) => w.name === "detail").value = 0.8;
        node.widgets.find((w) => w.name === "color").value = 0.6;
        node.widgets.find((w) => w.name === "quality").value = 18;
    });
    const stored = node.widgets.map((widget) => widget.value);
    assert.deepEqual(node.widgets.filter((w) => w.serialize !== false).map((widget) => widget.name), names);
    assert.ok(node.getLayoutWidgets().findIndex((w) => w.name === "detail")
        < node.getLayoutWidgets().findIndex((w) => w.name === "quality"));
    const restored = makeNode();
    restored.node.widgets.forEach((widget, index) => { widget.value = stored[index]; });
    restored.node.onConfigure({});
    assert.equal(restored.field("detail").value, 0.8);
    assert.equal(restored.field("color").value, 0.6);
    assert.equal(restored.field("quality").value, 18);
});

test("upstream controls reveal possibly active fields and disconnect restores visibility", () => {
    // A linked control's local value is stale; link id zero is valid.
    const { node, visible, change } = makeNode("MyVideoEnhance");
    node.inputs = [
        { name: "enable_neural_rendering", link: 0 },
        { name: "nr_profile", link: 1 },
        { name: "motion", link: 2 },
    ];
    change("motion", "none");
    node.onConnectionsChange(1, 0, true);
    assert.equal(visible("nr_profile"), true);
    assert.equal(visible("style"), true);
    assert.equal(visible("scene_cut_threshold"), true);
    node.inputs = [];
    node.onConnectionsChange(1, 0, false);
    assert.equal(visible("nr_profile"), false);
    assert.equal(visible("style"), false);
    assert.equal(visible("scene_cut_threshold"), false);
});

test("hiding preserves widget contracts and pre-existing hidden state", () => {
    // Prompt serialization and input conversion must retain their own methods.
    const serializer = function () { return this.value; };
    let originals;
    const { node, change, field, visible } = makeNode("MyVideoEnhanceStream", (node) => {
        node.widgets.find((w) => w.name === "detail").hidden = true;
        for (const widget of node.widgets) widget.serializeValue = serializer;
        originals = node.widgets.map((w) => [w, w.type, w.computeSize]);
    });
    change("enable_neural_rendering", true);
    assert.equal(visible("detail"), false);
    change("enable_neural_rendering", false);
    node.onGraphConfigured();
    for (const [widget, type, computeSize] of originals) {
        assert.equal(widget.type, type);
        assert.equal(widget.computeSize, computeSize);
        assert.equal(widget.serializeValue, serializer);
    }
    assert.equal(field("worker_timeout").serializeValue(), 600);
    assert.equal(node.size[0], 220);
    // A refresh without visibility changes must preserve manual node sizing.
    node.size[1] = 900;
    node.onConfigure({});
    assert.equal(node.size[1], 900);
});

test("unrelated node prototypes are untouched", () => {
    const nodeType = function () {};
    extension.beforeRegisterNodeDef(nodeType, { name: "OtherNode" });
    assert.equal(nodeType.prototype.onNodeCreated, undefined);
    assert.equal(nodeType.prototype.onConnectionsChange, undefined);
});

test("headings have no arrows or interaction; groups follow feature availability", () => {
    const { node, change, visible } = makeNode();
    change("enable_neural_rendering", true);
    change("nr_profile", "custom");
    assert.equal(visible("style"), true);
    assert.equal(visible("worker_timeout"), true);
    for (const heading of node.widgets.filter((w) => w.serialize === false)) {
        assert.equal(heading.callback, undefined);
        assert.equal(heading.mouse, undefined);
        assert.equal(heading.type, "video_enhance_heading");
        assert.doesNotMatch(heading.label, /[▼▶]/);
    }
    change("enable_neural_rendering", false);
    for (const name of ["style", "_ve_custom", "_ve_runtime", "worker_timeout"]) {
        assert.equal(visible(name), false);
    }
    assert.equal(visible("_ve_gimm"), true);
    change("enable_neural_rendering", true);
    assert.equal(visible("style"), true);
    assert.equal(visible("worker_timeout"), true);
});

test("UI headings never enter positional workflow values or API prompt inputs", () => {
    // Mirrors LiteGraph persistence and Comfy's separate prompt serialize flag.
    const { node, names } = makeNode();
    const originalWidgets = node.widgets.slice(0, names.length);
    const buttons = node.widgets.slice(names.length);
    assert.equal(buttons.length, 8);
    for (const button of buttons) {
        assert.equal(button.serialize, false);
        assert.equal(button.options.serialize, false);
    }
    const promptNames = node.widgets.filter((w) => w.options?.serialize !== false).map((w) => w.name);
    assert.deepEqual(promptNames, names);
    node.getLayoutWidgets();
    assert.deepEqual(node.widgets.slice(0, names.length), originalWidgets);
});
