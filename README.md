# Execute

```shell
docker run -it --rm -e input_dir=./input -e output_dir=./output -v "$PWD":/usr/src/app $(docker build -q .)
```