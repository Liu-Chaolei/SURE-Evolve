"""Route controller and research subprocesses through the local priority gateway."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--role', choices=['controller','xlab'], required=True)
    parser.add_argument('--router-config', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Command required')
    config = json.loads(args.router_config.read_text())
    token = Path(config['token_file']).read_text().strip()
    env = {key:value for key,value in os.environ.items() if not key.startswith(
        ('XI_','ZAI_','OPENAI_','LLM_','XLAB_RESEARCH_IDEA_'))}
    base = config['base_url']
    model = 'sure-priority-router'
    env.update(OPENAI_API_KEY=token,OPENAI_BASE_URL=base,ZAI_API_KEY=token,ZAI_BASE_URL=base,
               LLM_BASE_URL=base,LLM_MODEL=model,SURE_AGENT_MODEL=model)
    if args.role == 'xlab':
        env['XLAB_RESEARCH_IDEA_STREAM']='1'
        env['XLAB_RESEARCH_IDEA_TIMEOUT_SECONDS']='3000'
        for phase in ('AGENT','GENERATION','EVALUATION','FUSION'):
            env['XLAB_RESEARCH_IDEA_'+phase+'_MODEL']=model
    os.execvpe(command[0],command,env)


if __name__ == '__main__':
    main()
