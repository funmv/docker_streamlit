>activate py310a
>cd G:\2025\kafka\docker\crop_tool

>docker images
>docker build -t streamlit-crop_tool .
>docker run -p 8501:8501 -v G:/2025/kafka/docker/crop_tool:/app/data streamlit-crop_tool
>docker rmi -f image-id

>docker login
>docker tag streamlit-crop_tool:latest funmv/streamlit-crop_tool:latest
>docker push funmv/streamlit-crop_tool:latest
>docker pull funmv/streamlit-crop_tool:latest