find_open_port() {
    for port in {1024..49151}; do
        (echo > /dev/tcp/localhost/$port) >/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo $port
            return 0
        fi
    done
    echo "No open port found in the specified range." >&2
    exit 1
}