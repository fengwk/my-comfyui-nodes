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
    extension.beforeRegisterNodeDef(nodeType, { name });
    const names = [
        "enable_super_resolution", "spatial_mode", "enable_neural_rendering",
        "nr_profile", "nr_intensity", "enable_frame_interpolation",
        "output_codec", "quality", "vfi_precision", "motion",
        "scene_cut_threshold", "stage_order", "style", "detail", "color",
        "wine_prefix", "worker_timeout",
    ];
    const values = {
        enable_super_resolution: false, spatial_mode: "2.0x",
        enable_neural_rendering: false, nr_profile: "standard",
        nr_intensity: 1.0, enable_frame_interpolation: false,
        output_codec: "libx264", quality: 18, vfi_precision: "fp32",
        motion: "optical_flow", scene_cut_threshold: 0.2,
        stage_order: "vfi_then_dlss", style: "Cinematic",
        detail: 1, color: 1, wine_prefix: "", worker_timeout: 600,
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

test("only settings that apply are visible; output settings remain visible", () => {
    const { change, visible, field } = makeNode();
    assert.equal(visible("spatial_mode"), false);
    assert.equal(visible("nr_profile"), false);
    assert.equal(visible("vfi_precision"), false);
    assert.equal(visible("stage_order"), false);
    assert.equal(visible("output_codec"), true);
    assert.equal(field("quality").label, "Output · Quality");
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

test("saved widget positions and values survive hiding and workflow restore", () => {
    const { node, names, field, change, visible } = makeNode();
    const initial = node.widgets.map((widget) => widget.serializeValue?.() ?? widget.value);
    change("enable_neural_rendering", true);
    change("nr_profile", "custom");
    change("style", "Natural");
    change("nr_profile", "standard");
    assert.equal(visible("style"), false);
    assert.equal(field("style").serializeValue?.() ?? field("style").value, "Natural");
    assert.deepEqual(node.widgets.map((widget) => widget.name), names);
    assert.equal(initial.length, node.widgets.length);
    field("enable_neural_rendering").value = false;
    field("enable_super_resolution").value = true;
    node.onConfigure({});
    assert.equal(visible("spatial_mode"), true);
    assert.equal(visible("nr_profile"), false);
    assert.equal(field("style").value, "Natural");
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
