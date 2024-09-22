from sklearn.neural_network import MLPClassifier
import pickle
with open("risco_credito.pkl","rb") as f:
    x_credit_risk,y_credit_risk = pickle.load(f)

values_predicts = input("digite valores: ").lower()
history,debt,guarantee,income = values_predicts.split(",")

class Neural:
    def __init__(self,history=history,debt=debt,guarantee=guarantee,income=income,x_trainign = x_credit_risk,y_training = y_credit_risk) -> None:
        self._history = history
        self._debt = debt
        self._guarantee = guarantee
        self._income = income
        self._x = x_trainign
        self._y = y_training
        self.hard_compair = {"history":["bad","unknown","good"],
                            "debt":["high","low"],
                            "guarantee":["none","adequate"],
                            "income":["0_15","15_35","above_35"]}
        self.compair_values()

    
    def compair_values(self):
        for h in self.hard_compair["history"]:
            i = 0
            if self._history == h:
                self._history = i
            i += 1

        for d in self.hard_compair["debt"]:
            t = 0
            if self._debt == d:
                self._debt = t
            t += 1

        for g in self.hard_compair["guarantee"]:
            f= 0
            if self._guarantee == g:
                self._guarantee = f
            f += 1

        for r in self.hard_compair["income"]:
            b = 0
            if self._income == r:
                self._income = b
            b += 1

        
    
    def exec_rede(self):
        self._rede_neural = MLPClassifier(max_iter=500,hidden_layer_sizes=(3,3),tol=0.0000100)
        self._rede_neural.fit(self._x,self._y)

    def predic_rede(self):
        previsoes = self._rede_neural.predict([[self._history,self._debt,self._guarantee,self._income]])
        return previsoes
    


Neuro = Neural()
Neuro.exec_rede()
print(f"the risk of granting a loan to this person is {Neuro.predic_rede()}")