>activate py310a
>cd G:\2025\kafka\docker\load_tool

>docker images
>docker build -t streamlit-load-tool .
# 8502:8501로 할 때는 Dockerfile도 수정해야 함
>docker run -d --name streamlit-load-tool -p 8501:8501 -v "%CD%\output:/app/output" streamlit-load-tool
#background 실행이므로 제거할 때는 
>docker rm -f [container-id]

# 아래는 windows에서 실행 시에 문제 발생(old)
>docker run -p 8501:8501 -v G:/2025/kafka/docker/load_tool:/app/data streamlit-load-tool

>docker rmi -f image-id

>docker login
>docker tag streamlit-load_tool:latest funmv/streamlit-load-tool:latest
>docker push funmv/streamlit-load-tool:latest
>docker pull funmv/streamlit-load-tool:latest