# CS175

## Steps

### Docker Setup (Run in project terminal)
```
docker run -it -p 5901:5901 -p 6901:6901 -p 8888:8888 -e VNC_PW=password -v $(pwd):/home/malmo/cs175 andkram/malmo
```

### Connect to Docker Image Browser
```
http://localhost:6901/?password=password
```

### Connect to Jupyter Notebook
```
Look for output in the form

Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://0.0.0.0:8888/?token=1c6390221431ca75146946c52e253f063431b6488420bbac
```