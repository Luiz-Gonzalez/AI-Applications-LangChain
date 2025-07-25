from langchain_core.runnables import RunnableLambda

def cumprimentar(nome):
    return f'Olá {nome}!'

runnable_cumprimentar = RunnableLambda(cumprimentar)

resultado = runnable_cumprimentar.invoke('Luiz')
print(resultado)