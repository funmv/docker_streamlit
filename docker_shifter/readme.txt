>activate py310a
>cd G:\nox\2026\02_streamlit\01_shifter\02_docker_h5
>docker images
>docker build --no-cache -t streamlit-shifter .
>docker run -d --name streamlit-shifter -p 8501:8501 -v "%CD%\output:/app/output" streamlit-shifter

#>docker run -p 8501:8501 -v G:/nox/2025/extract_excel_busan#5GT:/app/data streamlit-viewer
#>docker run -p 8501:8501 -v %cd%:/app/data streamlit-viewer
#>docker run -p 8502:8501 -v %cd%:/app/data streamlit-viewer  # 포트가 사용 중일 때
>docker rmi -f image-id

>docker login
>docker tag streamlit-shifter:latest funmv/streamlit-shifter:latest
>docker push funmv/streamlit-shifter:latest
>docker pull funmv/streamlit-shifter:latest
>
####### ubuntu install guide ######
# --remove-orphans: Remove unused containers
###################################
> docker compose down
> docker compose up -d --build --remove-orphans  

