

def load_agent():
    with open('/nn/s_nn.json', 'r') as file:
        data = json.load(file)
    
    agent = Agent()
    agent.w1 = np.array(data['w1'])
    agent.w2 = np.array(data['w2'])
    agent.b1 = np.array(data['b1'])
    agent.b2 = np.array(data['b2'])
    
    return agent