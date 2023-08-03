from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from llama2_7b_4bit import LLAMa2_7b_4bitQ


app = Flask(__name__)
socketio = SocketIO(app)
print("Going to load the llama2 7b model ...")
llam2 = LLAMa2_7b_4bitQ()
print("Loaded LLama2 7b model")


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    print('received query: ' + data['input'])
    print('received system_message: ' + data['system_message'])
    prompt_template = llam2.create_prompt(data['input'],data['system_message'])
    try:
        output = llam2.generate_ouputput(prompt_template)
        response =llam2.get_chat_response(output)
        data = data['input'].replace('\n', '<br>')  # replace newline characters with <br> tags
        emit('response', {'query': data,'response': 'Response: '+ response}, broadcast=True)
    except:
        emit('response', {'query': data,'response': 'Sorry got an Error From Server: '}, broadcast=True)

if __name__ == '__main__':
    socketio.run(app)