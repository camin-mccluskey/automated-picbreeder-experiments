"""Jupyter controls and displays for the independent interpretability module."""

import base64
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import ipywidgets as widgets
import numpy as np

from .cppn import png_bytes
from .interpretability import Intervention, save_experiment, sweep, trace_network


def activation_pixels(values, limit):
    """Blue negative, white zero, red positive; fixed symmetric display scale."""
    normalized = np.clip(values / limit, -1, 1)
    white = 1 - np.abs(normalized)
    rgb = np.stack((white + np.maximum(normalized, 0), white,
                    white + np.maximum(-normalized, 0)), axis=-1)
    return (rgb * 255 + 0.5).astype(np.uint8)


def _image_html(pixels, label, width=150, detail=""):
    data = base64.b64encode(png_bytes(pixels)).decode()
    return (f'<div style="display:inline-block;vertical-align:top;margin:6px;max-width:230px">'
            f'<div>{escape(label)}</div><img width="{width}" src="data:image/png;base64,{data}" '
            f'alt="{escape(label)}" style="image-rendering:pixelated">'
            f'<div>{escape(detail)}</div></div>')


def node_label(key, config):
    labels = dict(zip(config.genome_config.input_keys,
                      ("Horizontal position", "Vertical position", "Distance from centre")))
    labels.update(dict(zip(config.genome_config.output_keys,
                           ("Hue output", "Saturation output", "Brightness output"))))
    labels.update(getattr(config, "node_labels", {}))
    return f"{labels.get(key, 'Hidden neuron')} ({key})"


def graph_svg(trace, config, limits, selected=None):
    """Layer the active DAG for display only; retain unused genes in a final column."""
    inputs = config.genome_config.input_keys
    labels = dict(zip(inputs, ("x", "y", "radius")))
    labels.update(dict(zip(config.genome_config.output_keys, ("H", "S", "B"))))
    labels.update({k: "bias" for k in inputs if k not in labels})
    depths = {key: 0 for key in inputs}
    remaining = set(trace.activations) - set(inputs)
    while remaining:
        ready = [key for key in remaining
                 if all(a in depths for a, b in trace.active_edges if b == key)]
        if not ready:
            raise ValueError("Active computation is not acyclic.")
        for key in ready:
            depths[key] = 1 + max((depths[a] for a, b in trace.active_edges if b == key), default=0)
        remaining.difference_update(ready)
    unused_depth = max(depths.values()) + 1
    depths.update({key: unused_depth for key in trace.genome.nodes if key not in depths})
    columns = {}
    for key, depth in depths.items():
        columns.setdefault(depth, []).append(key)
    positions = {key: (25 + depth * 215, 35 + row * 145)
                 for depth, keys in columns.items() for row, key in enumerate(sorted(keys))}
    width = max(x for x, y in positions.values()) + 175
    height = max(y for x, y in positions.values()) + 140
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'role="img" aria-label="CPPN computation graph">',
             '<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" '
             'orient="auto"><path d="M0,0 L6,3 L0,6" fill="#555"/></marker></defs>']
    for (a, b), connection in trace.genome.connections.items():
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        active = (a, b) in trace.active_edges
        colour = "#b53e36" if connection.weight >= 0 else "#356ba0"
        if not active:
            colour = "#bbb"
        dashed = 'stroke-dasharray="4 4"' if not connection.enabled else ""
        parts.append(f'<path d="M{x1+155},{y1+55} C{x1+185},{y1+55} {x2-30},{y2+55} {x2},{y2+55}" '
                     f'fill="none" stroke="{colour}" {dashed} marker-end="url(#arrow)">'
                     f'<title>{a} → {b}: w={connection.weight:.6g}; '
                     f'{"disabled" if not connection.enabled else "active" if active else "unused"}</title></path>')
        if active:
            parts.append(f'<text x="{(x1+155+x2)/2}" y="{(y1+y2)/2+50}" '
                         f'font-size="10" fill="{colour}">{connection.weight:.3g}</text>')
    for key, (x, y) in positions.items():
        active = key in trace.activations
        colour = "#a84425" if key == selected else "#777" if active else "#ccc"
        name = labels.get(key, "hidden")
        activation = "input" if key in inputs else trace.genome.nodes[key].activation
        parts.append(f'<rect x="{x}" y="{y}" width="155" height="125" rx="5" '
                     f'fill="white" stroke="{colour}" stroke-width="{3 if key == selected else 1}"/>')
        parts.append(f'<text x="{x+7}" y="{y+17}" font-size="12">{key}: {name} · {escape(activation)}</text>')
        if active:
            values = trace.activations[key]
            data = base64.b64encode(png_bytes(activation_pixels(values, limits[key]))).decode()
            parts.append(f'<image x="{x+7}" y="{y+26}" width="75" height="75" '
                         f'href="data:image/png;base64,{data}"/>')
            parts.append(f'<text x="{x+7}" y="{y+116}" font-size="10">'
                         f'[{values.min():.3g}, {values.max():.3g}]</text>')
        else:
            parts.append(f'<text x="{x+7}" y="{y+65}" font-size="11">Not evaluated</text>')
    parts.append('</svg>')
    return ''.join(parts)


class CPPNInspector:
    """A single-network workspace. Controls always act on its original baseline."""

    def __init__(self, network, *, size=None, save_dir="runs/interpretability"):
        self.network = network
        self.size = network.size if size is None else size
        self.save_dir = Path(save_dir)
        self.baseline = trace_network(network, size=self.size)
        self.current = self.baseline
        self.sweep_results = []
        self._updating = False
        self.limits = {key: max(float(np.abs(values).max()), 1e-6)
                       for key, values in self.baseline.activations.items()}
        self.kind = widgets.Dropdown(options=[("Connection weight", "weight"), ("Node bias", "bias"),
                                              ("Disable connection", "disable"), ("Clamp activation", "clamp")],
                                     description="Intervention", style={"description_width": "initial"})
        self.target = widgets.Dropdown(description="Target", layout=widgets.Layout(width="440px"))
        self.value = widgets.FloatSlider(description="Value", min=-2, max=2, step=.02,
                                         continuous_update=False, readout_format=".3f",
                                         layout=widgets.Layout(width="500px"))
        self.low = widgets.FloatText(value=-1, description="Sweep from")
        self.high = widgets.FloatText(value=1, description="to")
        self.count = widgets.BoundedIntText(value=7, min=2, max=31, description="Samples")
        self.apply_button = widgets.Button(description="Apply intervention")
        self.reset_button = widgets.Button(description="Reset to original")
        self.sweep_button = widgets.Button(description="Run sweep")
        self.selection = widgets.HTML()
        self.graph = widgets.HTML(layout=widgets.Layout(overflow="auto", width="100%"))
        self.maps = widgets.HTML()
        self.outputs = widgets.HTML()
        self.strip = widgets.HTML(layout=widgets.Layout(overflow="auto"))
        self.status = widgets.HTML()
        self.observation = widgets.Textarea(description="Observation", placeholder="What changed? What persisted?",
                                            layout=widgets.Layout(width="650px"))
        self.save_button = widgets.Button(description="Save experiment")
        self.kind.observe(self._targets_changed, names="value")
        self.target.observe(self._parameter_changed, names="value")
        self.value.observe(self._apply, names="value")
        self.apply_button.on_click(self._apply)
        self.reset_button.on_click(self._reset)
        self.sweep_button.on_click(self._sweep)
        self.save_button.on_click(self._save)
        self.widget = widgets.VBox([
            widgets.HTML(f'<b>Genome {network.genome.key}</b> · {self.size} × {self.size} inspection. '
                         'Each intervention starts from the saved original.'),
            widgets.HBox([self.kind, self.target]), self.selection, self.value,
            widgets.HBox([self.apply_button, self.reset_button]), self.status,
            self.outputs, self.maps,
            widgets.HTML('Graph: red positive weights; blue negative; grey unused; dashed disabled. '
                         'Node maps: blue negative, white zero, red positive. '
                         'Scales stay fixed to each node’s baseline range; out-of-range values saturate.'),
            self.graph, widgets.HBox([self.low, self.high, self.count]), self.sweep_button, self.strip,
            self.observation, self.save_button,
        ])
        self._targets_changed()

    def _targets_changed(self, change=None):
        self._updating = True
        kind = self.kind.value
        if kind in ("weight", "disable"):
            options = [(f"{node_label(a, self.network.config)} → {node_label(b, self.network.config)}", (a, b))
                       for a, b in sorted(self.baseline.active_edges)]
        else:
            options = [(node_label(key, self.network.config), key)
                       for key in sorted(self.network.genome.nodes) if key in self.baseline.activations]
        self.target.options = options
        self.target.value = options[0][1] if options else None
        self._updating = False
        self._parameter_changed()

    def _parameter_changed(self, change=None):
        if self._updating:
            return
        self._updating = True
        kind, target = self.kind.value, self.target.value
        available = target is not None
        self.apply_button.disabled = not available
        self.sweep_button.disabled = not available or kind == "disable"
        self.value.disabled = not available or kind == "disable"
        if available:
            if kind in ("weight", "disable"):
                original = self.network.genome.connections[target].weight
            elif kind == "bias":
                original = self.network.genome.nodes[target].bias
            else:
                original = float(self.baseline.activations[target].mean())
            # Expand first so changing targets cannot temporarily violate slider bounds.
            low, high = original - 1, original + 1
            self.value.min = min(self.value.min, low)
            self.value.max = max(self.value.max, high)
            self.value.value = original
            self.value.min, self.value.max = low, high
            self.low.value, self.high.value = low, high
        self._updating = False
        self._reset()

    def _spec(self):
        return Intervention(self.kind.value, self.target.value,
                            None if self.kind.value == "disable" else self.value.value)

    def _apply(self, change=None):
        if self._updating or self.target.value is None:
            return
        try:
            self.current = trace_network(self.network, self._spec(), size=self.size)
            self.sweep_results = []
            self.strip.value = ""
            self.status.value = "Intervention applied to the original network."
            self._draw()
        except (ValueError, OverflowError) as error:
            self.status.value = f"<b>Could not apply:</b> {escape(str(error))}"

    def _reset(self, button=None):
        self.current = self.baseline
        self.sweep_results = []
        self.strip.value = ""
        self._updating = True
        target = self.target.value
        if target is not None:
            if self.kind.value in ("weight", "disable"):
                original = self.network.genome.connections[target].weight
            elif self.kind.value == "bias":
                original = self.network.genome.nodes[target].bias
            else:
                original = float(self.baseline.activations[target].mean())
            self.value.value = original
        self._updating = False
        self.status.value = ("Original network. Clamp value is a proposed constant; press Apply to clamp."
                             if self.kind.value == "clamp" else "Original network; no intervention applied.")
        self._draw()

    def _draw(self):
        self.outputs.value = ''.join([
            _image_html(self.baseline.rgb, "Original"), _image_html(self.current.rgb, "Modified"),
            _image_html(np.abs(self.current.rgb.astype(int) - self.baseline.rgb.astype(int)).astype(np.uint8),
                        "Absolute RGB difference (0–255)"),
        ])
        target = self.target.value
        if target is None:
            self.selection.value = "No active connections available for this intervention."
            self.maps.value = ""
            self.graph.value = graph_svg(self.current, self.network.config, self.limits)
            return
        kind = self.kind.value
        config = self.network.config
        maps = []

        def activation_map(trace, key, label, note=""):
            if key not in trace.activations:
                return f"<p>{escape(node_label(key, config))} is not evaluated after this intervention.</p>"
            values = trace.activations[key]
            detail = (f"Range [{values.min():.4g}, {values.max():.4g}]; "
                      f"colour scale ±{self.limits[key]:.4g}. {note}")
            return _image_html(activation_pixels(values, self.limits[key]), label, detail=detail)

        if kind in ("weight", "disable"):
            source, key = target
            original = self.network.genome.connections[target].weight
            connection = self.current.genome.connections[target]
            current = f"{connection.weight:.4g}" if connection.enabled else "disabled"
            self.selection.value = (
                f"<b>{escape(node_label(source, config))} → {escape(node_label(key, config))}</b><br>"
                f"Original weight: {original:.4g} → Current weight: {current}")
            maps.append(activation_map(self.baseline, source, f"Source: {node_label(source, config)}",
                                       "Unchanged by this weight." if kind == "weight" else "Original source activation."))
            explanation = ("This weight scales the source activation before it enters the destination neuron. "
                           if kind == "weight" else "Disabling this connection removes its contribution to the destination neuron. ")
        else:
            key = target
            if kind == "bias":
                detail = (f"Original bias: {self.network.genome.nodes[key].bias:.4g} → "
                          f"Current bias: {self.current.genome.nodes[key].bias:.4g}")
            else:
                detail = (f"Activation clamped to {self.current.intervention.value:.4g}"
                          if self.current.intervention is not None else "Original activation; no clamp applied.")
            self.selection.value = f"<b>{escape(node_label(key, config))}</b><br>{detail}"
            explanation = ""
        for name, trace in (("Original", self.baseline), ("Modified", self.current)):
            maps.append(activation_map(trace, key, f"{node_label(key, config)}: {name.lower()}"))
        self.maps.value = (''.join(maps) + f'<p>{explanation}'
                          'Map colours show numerical activation, not final image colour.</p>')
        self.graph.value = graph_svg(self.current, self.network.config, self.limits, key)

    def _sweep(self, button=None):
        try:
            if not np.isfinite([self.low.value, self.high.value]).all() or self.low.value >= self.high.value:
                raise ValueError("Sweep limits must be finite and increasing.")
            results = sweep(self.network, self._spec(), np.linspace(self.low.value, self.high.value,
                                                                    self.count.value), size=self.size)
            self.sweep_results = results
            self.strip.value = ''.join(_image_html(r.rgb, f"value = {r.intervention.value:.4g}", 110)
                                       for r in results)
            self.status.value = f"{len(results)} independent interventions. Save includes this sweep."
        except (ValueError, OverflowError) as error:
            self.status.value = f"<b>Could not sweep:</b> {escape(str(error))}"

    def _save(self, button=None):
        specs = [r.intervention for r in self.sweep_results]
        if self.current.intervention is not None and self.current.intervention not in specs:
            specs.insert(0, self.current.intervention)
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        try:
            path = save_experiment(self.save_dir / name, self.network, specs,
                                   observation=self.observation.value, size=self.size)
            self.status.value = f"Saved {len(specs)} interventions to {escape(str(path))}"
        except (OSError, ValueError, OverflowError) as error:
            self.status.value = f"<b>Could not save:</b> {escape(str(error))}"

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)
