group "default" {
  targets = ["worker", "server"]
}

target "worker" {
  target = "runtime"
  tags = ["insight-bridge-worker:latest"]
}

target "server" {
  target = "server"
  tags = ["insight-bridge-server:latest"]
}
