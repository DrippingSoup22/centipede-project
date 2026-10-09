# The job of one Runpod pod, started in the background over SSH by
# runpod_gpu.py on the laptop:
#
#   bash runpod_pod.sh <launch> <grace seconds> <lifetime seconds>
#
# It runs the session, cloud_session.py, in a virtual environment that keeps
# the image's PyTorch (the image's own Python refuses pip installs, PEP 668),
# with the session's output in the launch's console folder. The session's
# exit code goes to /root/session_exit, which the laptop's copier waits for
# before it copies the output and deletes the pod. As a backstop against
# paying for a forgotten pod, the pod deletes itself when the copier has not
# done so within the grace time after the session, and in any case at the end
# of its lifetime.

launch=$1
grace_seconds=$2
lifetime_seconds=$3
console=/root/output/runpod/$launch
mkdir -p "$console"
exec > "$console/job.txt" 2>&1

# An SSH session does not inherit the container's environment, which holds
# the pod's id and its pod-scoped API key.
container_value() { tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p"; }
pod_id=$(container_value RUNPOD_POD_ID)
pod_key=$(container_value RUNPOD_API_KEY)

delete_pod() {
    echo "$(date -u '+%H:%M') deleting the pod: $1"
    curl -sS -X DELETE -H "Authorization: Bearer $pod_key" \
        "https://api.runpod.io/v2/pods/$pod_id"
}

(sleep "$lifetime_seconds" && delete_pod "its lifetime is over") &

echo "$(date -u '+%H:%M') session starting on pod $pod_id"
python -m venv --system-site-packages /root/venv
/root/venv/bin/python -u /root/cloud_session.py > "$console/session.txt" 2>&1 &
echo $! > /root/session.pid
wait $!
echo $? > /root/session_exit
echo "$(date -u '+%H:%M') session ended with exit code $(cat /root/session_exit)"

sleep "$grace_seconds"
delete_pod "its output was not copied within the grace time"
