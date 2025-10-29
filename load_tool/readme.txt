>activate py310a
>cd G:\2025\kafka\docker\load_tool

>docker images
>docker build -t streamlit-load_tool .
>docker run -p 8501:8501 -v G:/2025/kafka/docker/load_tool:/app/data streamlit-load_tool
>docker rmi -f image-id

>docker login
>docker tag streamlit-load_tool:latest funmv/streamlit-load_tool:latest
>docker push funmv/streamlit-load_tool:latest
>docker pull funmv/streamlit-load_tool:latest