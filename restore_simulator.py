import codecs

with codecs.open('old_simulator.py', 'r', encoding='utf-16') as f:
    content = f.read()

# Make sure we import from app.engine
imports = 'from app.engine import DayResult, DaySimulator\n'
if imports not in content:
    idx = content.find('from app.models')
    if idx != -1:
        end_idx = content.find('\n', idx)
        content = content[:end_idx+1] + imports + content[end_idx+1:]

# Make sure SimulationAgentMode is an Enum
enum_def = '''from enum import Enum
SimulationAgentMode = Literal["baseline_policy", "llm_inference", "trained_rl"]

class SimulationAgentModeEnum(str, Enum):
    baseline_policy = "baseline_policy"
    llm_inference = "llm_inference"
    trained_rl = "trained_rl"

SimulationAgentMode = SimulationAgentModeEnum
'''
if 'class SimulationAgentModeEnum' not in content:
    content = content.replace('SimulationAgentMode = Literal["baseline_policy", "llm_inference", "trained_rl"]', enum_def)

with codecs.open('app/simulator.py', 'w', encoding='utf-8') as f:
    f.write(content)
