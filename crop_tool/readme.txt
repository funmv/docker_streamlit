>activate py310a
>cd G:\2025\kafka\docker\crop_tool

>docker images
>docker build -t streamlit-crop-tool .
# 8502:8501로 할 때는 Dockerfile도 수정해야 함
>docker run -d --name streamlit-crop-tool -p 8501:8501 -v "%CD%\output:/app/output" streamlit-crop-tool
#background 실행이므로 제거할 때는 
>docker rm -f [container-id]

# 아래는 오류 발생(old)
>docker run -p 8501:8501 -v G:/2025/kafka/docker/crop_tool:/app/data streamlit-crop-tool

>docker rmi -f image-id

>docker login
>docker tag streamlit-crop-tool:latest funmv/streamlit-crop-tool:latest
>docker push funmv/streamlit-crop-tool:latest
>docker pull funmv/streamlit-crop-tool:latest


####### ubuntu install guide ######
# --remove-orphans: Remove unused containers
###################################
> docker compose down
> docker compose up -d --build --remove-orphans  