"""A notebook image selector, separated from the CPPN implementation."""

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import traceback

from IPython.display import display
import ipywidgets as widgets

from .breeding import BreedingSession
from .cppn import png_bytes, render
from .persistence import SessionWriter


class CPPNPlayground(BreedingSession):
    """Select one parent, retain it, and display eight mutated offspring."""

    def __init__(self, seed=7, size=96, save_dir="runs"):
        super().__init__(seed=seed)
        self.size = size
        self.save_dir = Path(save_dir)
        self.images = {}

        self.status = widgets.HTML()
        self.message = widgets.HTML()
        self.errors = widgets.Output()
        self.strength = widgets.FloatSlider(
            value=0.2, min=0.0, max=1.0, step=0.05, description="Mutation σ",
            continuous_update=False, style={"description_width": "initial"},
        )
        self.topology = widgets.Checkbox(value=True, description="Allow structure / activation changes",
                                         indent=False, layout=widgets.Layout(width="310px"))
        self.evolve_button = widgets.Button(description="Mutate selected", button_style="primary", disabled=True)
        self.back_button = widgets.Button(description="Back", disabled=True)
        self.reset_button = widgets.Button(description="New random images")
        self.save_button = widgets.Button(description="Save session")
        self.select_buttons = [widgets.Button(description=f"Select {i + 1}",
                                               layout=widgets.Layout(width="100%")) for i in range(9)]
        self.image_widgets = [widgets.Image(format="png", width=160, height=160) for _ in range(9)]
        self.labels = [widgets.HTML() for _ in range(9)]
        self.cards = [widgets.VBox([img, label, button], layout=widgets.Layout(
            padding="6px", border="2px solid transparent", align_items="center"))
            for img, label, button in zip(self.image_widgets, self.labels, self.select_buttons)]
        grid = widgets.GridBox(self.cards, layout=widgets.Layout(
            grid_template_columns="repeat(3, minmax(170px, 1fr))", grid_gap="8px",
            width="100%", max_width="620px"))
        self.widget = widgets.VBox([
            widgets.HTML("<b>CPPN image selection</b><br>Select a candidate below, then mutate it. "
                         "The parent stays in position 1; positions 2–9 are offspring."),
            widgets.HBox([self.strength, self.topology], layout=widgets.Layout(flex_flow="row wrap")),
            widgets.HBox([self.evolve_button, self.back_button, self.reset_button, self.save_button],
                         layout=widgets.Layout(flex_flow="row wrap")),
            self.status, grid, self.message, self.errors,
        ])
        for index, button in enumerate(self.select_buttons):
            button.on_click(lambda _, i=index: self._action(lambda: self.select(i)))
        self.evolve_button.on_click(lambda _: self._action(self.evolve))
        self.back_button.on_click(lambda _: self._action(self.back))
        self.reset_button.on_click(lambda _: self._action(self.reset))
        self.save_button.on_click(lambda _: self._action(self.save))
        self._show()

    def _action(self, callback):
        self.errors.clear_output()
        self.message.value = ""
        self.evolve_button.disabled = True
        try:
            callback()
        except Exception:
            with self.errors:
                traceback.print_exc()
        finally:
            self.evolve_button.disabled = self.selected_id is None

    def _show(self):
        for i, key in enumerate(self.candidates):
            genome = self.genomes[key]
            if key not in self.images:
                self.images[key] = png_bytes(render(genome, self.config, self.size))
            self.image_widgets[i].value = self.images[key]
            hidden = len(genome.nodes) - len(self.config.genome_config.output_keys)
            edges = sum(c.enabled for c in genome.connections.values())
            self.labels[i].value = f"<small>#{key} · {hidden} hidden · {edges} connections</small>"
        self._mark_selection()
        self.back_button.disabled = not self.history

    def _mark_selection(self):
        for i, key in enumerate(self.candidates):
            selected = key == self.selected_id
            self.cards[i].layout.border = "2px solid #3573b9" if selected else "2px solid transparent"
            self.select_buttons[i].description = "Selected" if selected else f"Select {i + 1}"
            self.select_buttons[i].button_style = "info" if selected else ""
        selection = f"Selected: #{self.selected_id}" if self.selected_id is not None else "Choose a parent."
        self.status.value = f"Round {self.round} · {len(self.genomes)} candidates created · {selection}"
        self.evolve_button.disabled = self.selected_id is None

    def reset(self):
        super().reset()
        self._show()

    def select(self, position):
        super().select(position)
        self._mark_selection()

    def evolve(self):
        super().evolve(strength=self.strength.value, topology=self.topology.value)
        self._show()

    def back(self):
        super().back()
        self._show()

    def save(self):
        """Export the shared session format and every generated image."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ")
        directory = self.save_dir / f"cppn-{stamp}"
        writer = SessionWriter(directory, size=self.size, selector={"name": "human"})
        writer.save(self, images=self.images)
        self.message.value = f"Saved: <code>{escape(str(directory.resolve()))}</code>"
        return directory

    def display(self):
        display(self.widget)
