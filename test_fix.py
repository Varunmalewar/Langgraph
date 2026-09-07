"""Standalone validation of the fix for 4_parallel_workflowwithoutllm.ipynb.

Root cause of InvalidUpdateError: parallel (fan-out) nodes returned the WHOLE
state, so three nodes wrote `runs` (and other inputs) in the same superstep and
the default LastValue channel can accept only one value per step.

Fix: each parallel node returns ONLY the single key it computes. The summary
node returns a dict (a valid state update) instead of a raw str.
"""
from typing import TypedDict, NotRequired
from langgraph.graph import StateGraph, START, END


class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    strike_rate: NotRequired[float]
    bpb: NotRequired[float]
    boundary_percentage: NotRequired[float]
    summary: NotRequired[str]


def calculate_sr(state: BatsmanState) -> dict:
    strike_rate = (state["runs"] / state["balls"]) * 100
    return {"strike_rate": strike_rate}


def calculate_bpb(state: BatsmanState) -> dict:
    bpb = state["balls"] / (state["fours"] + state["sixes"])
    return {"bpb": bpb}


def calculate_boundary_percentage(state: BatsmanState) -> dict:
    boundary_percentage = ((state["fours"] * 4 + state["sixes"] * 6) / state["balls"]) * 100
    return {"boundary_percentage": boundary_percentage}


def summary(state: BatsmanState) -> dict:
    text = (
        f"Runs: {state['runs']}, Balls: {state['balls']}, Fours: {state['fours']}, "
        f"Sixes: {state['sixes']}, Strike Rate: {state.get('strike_rate', 'N/A')}, "
        f"Balls per Boundary: {state.get('bpb', 'N/A')}, "
        f"Boundary Percentage: {state.get('boundary_percentage', 'N/A')}"
    )
    return {"summary": text}


graph = StateGraph(BatsmanState)

graph.add_node("calculate_sr", calculate_sr)
graph.add_node("calculate_bpb", calculate_bpb)
graph.add_node("calculate_boundary_percentage", calculate_boundary_percentage)
graph.add_node("summary", summary)


graph.add_edge(START, "calculate_sr")
graph.add_edge(START, "calculate_bpb")
graph.add_edge(START, "calculate_boundary_percentage")

# canonical fan-in / barrier: summary runs once after all three complete
graph.add_edge(["calculate_sr", "calculate_bpb", "calculate_boundary_percentage"], "summary")

graph.add_edge("summary", END)


workflow = graph.compile()

final_state = workflow.invoke({"runs": 100, "balls": 60, "fours": 10, "sixes": 5})

print("SUCCESS - no InvalidUpdateError")
print("final_state =", final_state)
