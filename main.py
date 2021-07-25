import csv
from agent import Agent
import keras
from keras.models import load_model
import sys
import matplotlib.pyplot as plt
import numpy as np
import math

def formatPrice(n):#to print prices in a appropriate form
    x = float(n)
    if x < 0:
        return ("-$" + "{0:.2f}".format(abs(x)))
    else:
        return ("$" + "{0:.2f}".format(abs(x)))

def getStockDataVec(key):#to read in as float vec
    vec = []
    lines = open(key + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec

def sigmoid(x):#sigmoid function
    return 1 / (1 + math.exp(-x))

def getState(data, t, n):
    d = t - n + 1                            #starting time d
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t
    res = []
    for i in range(n - 1):
        res.append(sigmoid(float(block[i + 1]) - float(block[i])))
    return np.array([res])

def readData():
    close_price=[] #to store the data from the csv downloaded
    high_price=[]
    open_price=[]
    low_price=[]
    dates =[]
    with open("PBR.csv") as csvfile:
        next(csvfile)            #move to the second line of the csv file
        reader=csv.reader(csvfile)
        for row in reader:
            date=row[0]
            dates.append(date)  #dates
            close_p=row[5]
            close_price.append(close_p)
        print(dates)
        return close_price,dates


def main():
    global float
    windows=10
    episode=20
    agent=Agent(windows)#self does not require initialize, model name will only be used in test model
    close_prices,dates=readData()
    cp=[]
    #print(close_prices)
    #print(close_prices[0])
    #print('test')
    #data = getStockDataVec('AAPLtest')
    #print(data)
    for x in range(len(close_prices)-1):
        cp.append(float(close_prices[x]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(cp)                           #to show the pic of the stock
    plt.show()
    #print(dates)
    train_length=len(close_prices)-1
    batch_size=32
    for e_count in range(episode+1):
        print('Episode:'+str(e_count)+'/'+str(episode))
        state=getState(close_prices,0,windows+1)
        total_profit=0
        agent.inventory=[]              #inventory of stocks
        for i_count in range(train_length):
            action=agent.act(state)
            op=agent.option(state)

            print('options are'+str(op))
            print('actions are'+str(action))
            next_state=getState(close_prices,i_count+1,windows+1)
            reward=0
            print('epsilon')
            print(agent.epsilon)
            if action == 1:          # to buy the stock
                agent.inventory.append(close_prices[i_count])
                print(dates[i_count])
                print('Train: Buy at current price of '+formatPrice(close_prices[i_count]))

            elif action == 2 and len(agent.inventory) > 0:  # sell
                reward=0
                for fcount in range(len(agent.inventory)-1):
                    bought_price = float(agent.inventory.pop(0))
                    reward += float(close_prices[i_count]) - bought_price # sum all the rewards of sell

                total_profit += reward
                print(dates[i_count])
                print('Train: Sell at current price of ' + formatPrice(close_prices[i_count]) + " | Made a profit of " + formatPrice(reward))
            elif action==0:   # we give a negative reward for holdong stocks otherwise the model will be trained to be very discreeet
                reward=-1

            done = True if i_count == train_length - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state #move to the next time step
            if done:
                print('Train Total Profit: ' + formatPrice(total_profit))

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if e_count % 2 == 0:
            print("ecount"+str(e_count))
            agent.model.save("models/model_p" + str(e_count))
            model_name = "model_p" + str(e_count)
            model_e = load_model("models/" + model_name)
            window_size = model_e.layers[0].input.shape.as_list()[1]
            agent_e = Agent(window_size, True, model_name)
            data = getStockDataVec('PBR(1)')
            lx = len(data)
            state_e = getState(data, 0, window_size + 1)
            total_profit_e = 0
            agent_e.inventory = []
            action_list=[]
            result=[]
            for tt in range(lx - 1):
                action_e = agent.act(state_e)
                action_list.append(action_e)
                next_state_e = getState(data, tt + 1, window_size + 1)
                reward_e = 0
                if action_e == 1:  # buy
                    agent_e.inventory.append(data[tt])#buy the stock at tt times
                    print('Test Buy: ' + formatPrice(data[tt]))
                elif action_e == 2 and len(agent_e.inventory) > 0:  # sell all the stocks
                    reward_e=0
                    for qcount in range(len(agent_e.inventory)-1):
                        bought_price_e = float(agent_e.inventory.pop(0))
                        reward_e += float(data[tt]) - bought_price_e


                    total_profit_e += reward_e #get the total profit
                    print('Test Sell: ' + formatPrice(data[tt]) + '|Profit: ' + formatPrice(reward_e))
                done_e=True if tt==lx-2 else False
                agent_e.memory.append((state_e,action_e,reward_e,next_state_e,done_e))

                state_e=next_state_e #move to the next time step
                if done_e:
                    print('e_count'+str(e_count))
                    print('Total test profit is'+format(total_profit_e))
                    result.append(total_profit_e)
            print(result)
            print('action list is'+str(action_list))



if __name__ == "__main__": #run the main function
    main()


'''
    #test the function
    start=train_length+1
    test_total_profit = 0
    agent.inventory = []
    for t in range(len(close_prices)-start):
        time=i+start
        test_state = getState(close_prices, start, windows + 1)
        action_test = agent.act(test_state)
        next_state = getState(close_prices, time + 1, windows + 1)
        reward = 0
        if action_test == 1:
            agent.inventory.append(close_prices[time])
            print(dates[time])
            print('Real: Buy at current price of ' + formatPrice(close_prices[time]))

        elif action_test == 2 and len(agent.inventory) > 0:  # sell
            bought_price_s = agent.inventory.pop(0)
            bought_price = float(bought_price_s)
            reward = max(float(close_prices[time]) - bought_price, 0)
            test_total_profit += float(close_prices[time]) - bought_price
            print(dates[time])
            print(
                'Real: Sell at current price of ' + formatPrice(close_prices[time]) + " | Made a profit of " + formatPrice(
                    float(close_prices[time]) - bought_price))

        done = True if t == len(close_prices)-start else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print('--------------------------------')
            print('Real!!Total Profit is' + formatPrice(test_total_profit))

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

'''

