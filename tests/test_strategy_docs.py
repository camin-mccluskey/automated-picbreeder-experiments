"""Keep strategy guides discoverable and their CLI examples/options usable offline."""

from pathlib import Path
import inspect
import re
import shlex

import pytest

from automated_picbreeder.selection_cli import COMMANDS, build_parser
from automated_picbreeder import selection_strategies


ROOT = Path(__file__).resolve().parents[1]
GUIDES = ROOT / "docs" / "strategies"
COMMON = (ROOT / "docs" / "run-options.md", GUIDES / "options.md")
DOCS = (ROOT / "README.md", *(path for path in sorted((ROOT / "docs").rglob("*.md"))
                             if not any(part.startswith(".") for part in path.relative_to(ROOT).parts)),
        ROOT / "src" / "automated_picbreeder" / "reference_data" / "skull" / "README.md")
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _commands(path):
    for block in re.findall(r"```sh\n(.*?)```", path.read_text(), re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            tokens = shlex.split(line, comments=True)
            if "experiments/run_selection.py" in tokens:
                yield tokens[tokens.index("experiments/run_selection.py") + 1:]


@pytest.mark.parametrize("name", [name for name, _, _ in COMMANDS])
def test_each_strategy_has_linked_guide_covering_every_cli_option(name):
    guide = GUIDES / f"{name}.md"
    assert f"docs/strategies/{name}.md" in (ROOT / "README.md").read_text()
    assert guide.exists()
    text = guide.read_text() + "\n" + COMMON[0].read_text()
    shared = COMMON[1].read_text()
    for anchor in re.findall(r"options\.md#([\w-]+)", guide.read_text()):
        for section in re.split(r"(?=^## )", shared, flags=re.MULTILINE):
            heading = section.splitlines()[0].removeprefix("## ").lower().replace(" ", "-")
            if heading == anchor:
                text += "\n" + section
    parser = build_parser().parse_args([name]).command_parser
    for action in parser._actions:
        for option in action.option_strings:
            if option.startswith("--"):
                assert f"`{option}`" in text, f"Undocumented option for {name}: {option}"
    assert any(args[0] == name and "--help" not in args for args in _commands(guide))


@pytest.mark.parametrize("name", [name for name, _, _ in COMMANDS])
def test_documented_python_constructor_defaults_match_source(name):
    text = (GUIDES / f"{name}.md").read_text()
    constructor = re.search(r"\b(\w+SelectionStrategy)\(", text).group(1)
    signature = inspect.signature(getattr(selection_strategies, constructor))
    compact = re.sub(r"\s+", "", text)
    for parameter in signature.parameters.values():
        assert parameter.default is not inspect.Parameter.empty
        value = repr(parameter.default).replace("'", '\"')
        assert re.sub(r"\s+", "", f"{parameter.name}={value}") in compact, (
            f"Update {name}'s documented default: {parameter.name}={value}"
        )


@pytest.mark.parametrize("path", DOCS, ids=lambda path: str(path.relative_to(ROOT)))
def test_documented_selection_commands_parse_without_loading_models(path):
    for args in _commands(path):
        if "--help" in args or "-h" in args:
            with pytest.raises(SystemExit) as error:
                build_parser().parse_args(args)
            assert error.value.code == 0
        else:
            build_parser().parse_args(args)


@pytest.mark.parametrize("path", DOCS, ids=lambda path: str(path.relative_to(ROOT)))
def test_documentation_local_links_and_anchors_exist(path):
    for link in LINK.findall(path.read_text()):
        if "://" in link or link.startswith("mailto:"):
            continue
        filename, _, anchor = link.partition("#")
        target = (path.parent / filename).resolve() if filename else path
        assert target.exists(), f"Broken link in {path.relative_to(ROOT)}: {link}"
        if anchor and target.suffix == ".md":
            headings = re.findall(r"^#+\s+(.+)$", target.read_text(), re.MULTILINE)
            slugs = {re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
                     for heading in headings}
            assert anchor in slugs, f"Broken heading in {path.relative_to(ROOT)}: {link}"
