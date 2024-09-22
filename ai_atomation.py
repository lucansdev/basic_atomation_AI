from sklearn.neural_network import MLPClassifier
import pickle
with open("risco_credito.pkl","rb") as f:
    x_risco_credito,y_risco_credito = pickle.load(f)

values_predicts = input("digite valores: ").lower()
historia,divida,garantia,renda = values_predicts.split(",")

class RedeNeural:
    def __init__(self,historia=historia,divida=divida,garantia=garantia,renda=renda,x_trainign = x_risco_credito,y_training = y_risco_credito) -> None:
        self._historia = historia
        self._divida = divida
        self._garantia = garantia
        self._renda = renda
        self._x = x_trainign
        self._y = y_training
        self.hard_compair = {"historia":["ruim","desconhecida","boa"],
                            "divida":["alta","baixa"],
                            "garantia":["nenhuma","adequada"],
                            "renda":["0_15","15_35","acima_35"]}
        self.compair_values()

    
    def compair_values(self):
        for h in self.hard_compair["historia"]:
            i = 0
            if self._historia == h:
                self._historia = i
            i += 1

        for d in self.hard_compair["divida"]:
            t = 0
            if self._divida == d:
                self._divida = t
            t += 1

        for g in self.hard_compair["garantia"]:
            f= 0
            if self._garantia == g:
                self._garantia = f
            f += 1

        for r in self.hard_compair["renda"]:
            b = 0
            if self._renda == r:
                self._renda = b
            b += 1

        
    
    def exec_rede(self):
        self._rede_neural = MLPClassifier(max_iter=500,hidden_layer_sizes=(3,3),tol=0.0000100)
        self._rede_neural.fit(self._x,self._y)

    def predic_rede(self):
        previsoes = self._rede_neural.predict([[self._historia,self._divida,self._garantia,self._renda]])
        return previsoes
    


teste_rede = RedeNeural()
teste_rede.exec_rede()
print(f"the risk of granting a loan to this person is {teste_rede.predic_rede()}")