import os

with open('api/index.py', 'r') as f:
    api_content = f.read()

metrics_part = api_content.split('@app.get("/metrics")')[1]

with open('api_server.py', 'r') as f:
    server_content = f.read()

new_server = server_content.replace('if __name__ == "__main__":', f'\n@app.get("/metrics")\n{metrics_part}\nif __name__ == "__main__":')

with open('api_server.py', 'w') as f:
    f.write(new_server)
print('Merged endpoints.')
