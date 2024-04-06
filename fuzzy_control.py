import numpy as np

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Antecedent
size_ratio = ctrl.Antecedent(np.linspace(0, .4, num=1001), 'size_ratio')
delta_loss = ctrl.Antecedent(np.linspace(-0.3, 0.3, num=1001), 'delta_loss')

#Consequent
alfa_adjustment = ctrl.Consequent(np.linspace(-0.2, 0.75, num=1001), 'alfa_adjustment') 

# Membership function definition
size_ratio['very_small'] = fuzz.trimf(size_ratio.universe, [0.0, 0.0, 0.03])
size_ratio['small'] = fuzz.trimf(size_ratio.universe, [0.0, 0.03, 0.1])
size_ratio['big'] = fuzz.trapmf(size_ratio.universe, [0.09, 0.20, 1.0, 1.0])

delta_loss['very_negative'] = fuzz.trapmf(delta_loss.universe, [-0.3, -0.3, -0.2, -0.03])
delta_loss['negative'] = fuzz.trimf(delta_loss.universe, [-0.07, -0.03, 0])
delta_loss['small_negative'] = fuzz.trimf(delta_loss.universe, [-0.03, 0, 0.0015])
delta_loss['positive'] = fuzz.trimf(delta_loss.universe, [0, 0.0015, 0.005])
delta_loss['very_positive'] = fuzz.trapmf(delta_loss.universe, [0.0015, 0.15, 0.35, 0.35])

alfa_adjustment['decrement'] = fuzz.trimf(alfa_adjustment.universe, [-0.2, -0.2, -0.05])
alfa_adjustment['slight_decrement'] = fuzz.trimf(alfa_adjustment.universe, [-0.1, -0.05, 0])
alfa_adjustment['null'] = fuzz.trimf(alfa_adjustment.universe, [-0.05, 0, 0.05])
alfa_adjustment['slight_increment'] = fuzz.trimf(alfa_adjustment.universe, [0, 0.25, 0.75])
alfa_adjustment['increment'] = fuzz.trimf(alfa_adjustment.universe, [0.25, 0.75, 0.75])

rules = []

rules.append(ctrl.Rule(size_ratio['very_small'] & delta_loss['very_negative'], alfa_adjustment['null']))
rules.append(ctrl.Rule(size_ratio['very_small'] & delta_loss['negative'], alfa_adjustment['null']))
rules.append(ctrl.Rule(size_ratio['very_small'] & delta_loss['small_negative'], alfa_adjustment['slight_increment']))
rules.append(ctrl.Rule(size_ratio['very_small'] & delta_loss['positive'], alfa_adjustment['increment']))
rules.append(ctrl.Rule(size_ratio['very_small'] & delta_loss['very_positive'], alfa_adjustment['increment']))

rules.append(ctrl.Rule(size_ratio['small'] & delta_loss['very_negative'], alfa_adjustment['null']))
rules.append(ctrl.Rule(size_ratio['small'] & delta_loss['negative'], alfa_adjustment['null']))
rules.append(ctrl.Rule(size_ratio['small'] & delta_loss['small_negative'], alfa_adjustment['null']))
rules.append(ctrl.Rule(size_ratio['small'] & delta_loss['positive'], alfa_adjustment['null']))
rules.append(ctrl.Rule(size_ratio['small'] & delta_loss['very_positive'], alfa_adjustment['slight_increment']))

rules.append(ctrl.Rule(size_ratio['big'] & delta_loss['very_negative'], alfa_adjustment['null']))
rules.append(ctrl.Rule(size_ratio['big'] & delta_loss['negative'], alfa_adjustment['null']))
rules.append(ctrl.Rule(size_ratio['big'] & delta_loss['small_negative'], alfa_adjustment['null']))
rules.append(ctrl.Rule(size_ratio['big'] & delta_loss['positive'], alfa_adjustment['slight_decrement']))
rules.append(ctrl.Rule(size_ratio['big'] & delta_loss['very_positive'], alfa_adjustment['slight_decrement']))


# Crear el sistema de control utilizando las reglas de inferencia
alfa_ctrl = ctrl.ControlSystem(rules)
alfa_ = ctrl.ControlSystemSimulation(alfa_ctrl)

#size_ratio.view()
#delta_loss.view()
#alfa_adjustment.view()

def act_alpha(size_ratio, focal_n0, focal_n1):
    
    delta = focal_n1-focal_n0
       
    alfa_.input['size_ratio'] = np.clip(size_ratio, 0, 1)
    alfa_.input['delta_loss'] = np.clip(delta, -0.3, 0.3)
    
    alfa_.compute()
    return alfa_.output['alfa_adjustment']
