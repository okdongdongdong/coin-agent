from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .base import SubAgent
from ..models.agent import AgentState

LOGGER = logging.getLogger(__name__)


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, SubAgent] = {}
        self._states: Dict[str, AgentState] = {}

    def register(self, agent: SubAgent) -> None:
        self._agents[agent.agent_id] = agent
        self._states[agent.agent_id] = AgentState(
            agent_id=agent.agent_id,
            strategy_name=agent.strategy_name(),
        )
        LOGGER.info("Registered agent: %s (%s)", agent.agent_id, agent.strategy_name())

    def get(self, agent_id: str) -> Optional[SubAgent]:
        return self._agents.get(agent_id)

    def get_state(self, agent_id: str) -> Optional[AgentState]:
        return self._states.get(agent_id)

    def set_state(self, agent_id: str, state: AgentState) -> None:
        self._states[agent_id] = state

    def active_agents(self) -> List[SubAgent]:
        return [
            a for a in self._agents.values()
            if self._states.get(a.agent_id, AgentState(a.agent_id, "")).is_active
        ]

    def all_agents(self) -> List[SubAgent]:
        return list(self._agents.values())

    def all_states(self) -> Dict[str, AgentState]:
        return dict(self._states)

    def agent_ids(self) -> List[str]:
        return list(self._agents.keys())

    def load_states(self, states: Dict[str, Dict]) -> None:
        for agent_id, data in states.items():
            if agent_id in self._agents:
                self._states[agent_id] = AgentState.from_dict(data)

    def dump_states(self) -> Dict[str, Dict]:
        return {aid: st.to_dict() for aid, st in self._states.items()}
