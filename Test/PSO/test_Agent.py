from unittest import TestCase

from PSO.Agent.Agent import Agent
from PSO.Environment.ObjectiveFunctions.Testing.Functions.SineEnvelope import SineEnvelopeFunction

class TestAgent(TestCase):

    def test_init(self):
        agent = Agent()

        assert agent.number_dimensions == 3

        agent.load_environment(
            SineEnvelopeFunction()
        )

        print(
            agent.step(1,1,1)
        )
