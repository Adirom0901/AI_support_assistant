from graph import graph
from langchain.messages import HumanMessage


# Due to invoke, the state given to graph starts from scratch so although we're mainting a 
# state, the memory is re-initialised with each invoke.
# added checkpointer in graph.py and config & thread_id here
thread_id='2'
while True:
    user_message= input('Type here: ')

    if user_message.strip().lower() in ['exit','quit','bye']:
        break
    config= {'configurable':{'thread_id':thread_id}}
    result = graph.invoke({
        'messages': [HumanMessage(content= user_message)]
    }
    , config=config)
    print(result['messages'][-1].content)
    pass




