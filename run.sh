
critic=CriticHierarchical
seed=$1
gpu=$2
agent=$3
env=$4
trianglescale=$5
hlactornum=$6
USE_CAPABLE=$7
USE_MULTI_LL=$8
USE_VISUALIZE=$9
dimhidden=256
lr=0.001




epochs=100

if [ "$env" = "FetchReach" ]; then
  # FetchReach のときは 10 に
  epochs=10
elif [ "$env" = "FetchPush" ]; then
  epochs=20
elif [ "$env" = "FetchPick" ]; then
  epochs=50
elif [ "$env" = "FetchSlide" ]; then
  epochs=50
elif [ "$env" = "PointMaze_Open_Diverse_GR" ]; then
  epochs=50
elif [ "$env" = "HandManipulateEggRotate" ]; then
  epochs=50
elif [ "$env" = "HandManipulateBlockRotateZ" ]; then
  epochs=50
fi



# 追加フラグを配列で管理
extra_flags_array=()
if [[ "$env" != *HandManipulate* && "$USE_VISUALIZE" == "true" ]]; then
    extra_flags_array+=(--visualize-flag)
fi

if [ "$USE_CAPABLE" = "true" ]; then
    extra_flags_array+=(--actor-capable)
fi

if [ "$USE_MULTI_LL" = "true" ]; then
    extra_flags_array+=(--multi_ll)
fi



export CUDA_VISIBLE_DEVICES=$gpu && MUJOCO_GL=egl python -u main.py --env-name=$env --lr-actor $lr --lr-critic $lr --n-epochs $epochs --agent $agent --negative-reward --critic $critic --seed $seed --cuda --dim-hierarchical-latent $dimhidden --triangle-scale $trianglescale --adversarial-flag --dim-prm-hidden $dimhidden --dim-critic-hidden $dimhidden --dim-hidden $dimhidden --high-level-actor-num $hlactornum  "${extra_flags_array[@]}"

