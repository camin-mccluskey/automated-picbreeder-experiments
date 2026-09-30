"""Interpret saved display visits, selection clicks and actual genome ancestry."""


def session_history(data):
    records = {genome["key"]: genome for genome in data["genomes"]}
    displays, selections, origins = [], [], {}
    preceding_actions = []
    selected = None
    for index, event in enumerate(data["events"]):
        action = event["action"]
        if action in {"reset", "evolve", "back"}:
            ids = tuple(event["displayed"])
            if len(ids) != 9:
                raise ValueError("Expected nine candidates per display.")
            if action == "reset":
                origins[ids] = {"round": 0, "retained_parent": None, "mutation": None}
                selected = None
            elif action == "evolve":
                parent = event["parent"]
                if ids[0] != parent or any(records[key]["parent"] != parent for key in ids[1:]):
                    raise ValueError("Candidate ancestry does not match the saved breeding event.")
                origins[ids] = {"round": displays[-1]["round"] + 1, "retained_parent": parent,
                                "mutation": {"strength": event["strength"], "topology": event["topology"]}}
                selected = parent
            else:
                if ids not in origins:
                    raise ValueError("Back refers to a grid that was never displayed.")
                selected = event["selected"]
            displays.append({"display_index": len(displays), "event_index": index, "action": action,
                             "displayed": list(ids), **origins[ids], "selected_id": selected,
                             "selection_generations": [],
                             "candidates": [{"genome_id": key, "image": records[key]["image"]} for key in ids]})
            preceding_actions.append(action)
        elif action == "select":
            display = displays[-1]
            if event["displayed"] != display["displayed"] or event["genome"] != event["displayed"][event["position"]]:
                raise ValueError("Selection does not match its displayed grid.")
            selections.append({"event": event, "event_index": index, "display_index": display["display_index"],
                               "round": display["round"], "retained_parent": display["retained_parent"],
                               "mutation": display["mutation"], "preceding_actions": preceding_actions,
                               "candidate_presentations": 9*len(displays)})
            display["selection_generations"].append(len(selections) - 1)
            selected = display["selected_id"] = event["genome"]
            preceding_actions = []
        else:
            raise ValueError(f"Unknown session event: {action}")
    if selected != data["selected"]:
        raise ValueError("Final selection does not match the saved event history.")
    ancestry, visited = [], set()
    key = selected
    while key is not None:
        if key in visited:
            raise ValueError("Cycle in genome ancestry.")
        visited.add(key)
        record = records[key]
        ancestry.append({"genome_id": key, "parent_id": record["parent"], "image": record["image"],
                         "selection_generations": [i for i, item in enumerate(selections) if item["event"]["genome"] == key]})
        key = record["parent"]
    ancestry.reverse()
    return selections, displays, ancestry
