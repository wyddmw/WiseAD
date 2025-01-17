export CARLA_ROOT=${1:-/home/spyder/project/LLMs/MobileVLM_Drive/carla}
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

# export SRC_ROOT=${2:-/home/spyder/project/LLMs/LingoQA_MobileVLM}
export SRC_ROOT='/home/spyder/project/e2e_driving/LingoQA_MobileVLM'
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/mobilevlm
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/mobilevlm/model
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/leaderboard
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:${SRC_ROOT}/scenario_runner
export LEADERBOARD_ROOT=${SRC_ROOT}/leaderboard
# export ROUTES=${SRC_ROOT}/leaderboard/data/validation_routes/routes_town07_short.xml
export ROUTES=${SRC_ROOT}/leaderboard/data/validation_routes/routes_town05_tiny.xml
# export ROUTES=${SRC_ROOT}/leaderboard/data/validation_routes/routes_town05_short.xml

export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000 # same as the carla server port #2000
export TM_PORT=8000 # port for traffic manager, required when spawning multiple servers/clients #2500
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs

export TEAM_AGENT=${SRC_ROOT}/leaderboard/team_code/llm_agent.py # agent
export TEAM_CONFIG=${SRC_ROOT}/leaderboard/team_code/corl_config.py
export CHECKPOINT_ENDPOINT=${SRC_ROOT}/results/town05_tiny_joint_epoch1_second.json # results file
# export SCENARIOS=${SRC_ROOT}/leaderboard/data/scenarios/town07_all_scenarios.json
export SCENARIOS=${SRC_ROOT}/leaderboard/data/scenarios/town05_all_scenarios.json

#export ROUTES=${SRC_ROOT}/leaderboard/data/42routes/42routes.xml #validation_routes/routes_town04_sample.xml
export SAVE_PATH=${SRC_ROOT}/data/eval # path for saving episodes while evaluating
export RESUME=False #True

# python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
python ${SRC_ROOT}/leaderboard/leaderboard/evaluation.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}
