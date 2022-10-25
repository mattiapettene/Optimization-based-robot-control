#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:53:57 2021

@author: adelprete
"""
import numpy as np
from numpy.linalg import norm, solve


def rk1(x, h, u, t, ode, jacobian=False):
    if(jacobian==False):
        dx = ode.f(x, u, t)
        x_next = x + h*dx
        return x_next, dx

    (f, f_x, f_u) = ode.f(x, u, t, jacobian=True)
    dx = f
    x_next = x + h*f
    
    nx = x.shape[0]
    I = np.identity(nx)    
    phi_x = I + h*f_x
    phi_u = h * f_u
    return x_next, dx, phi_x, phi_u

def rk2(x, h, u, t, ode):
    k1 = ode.f(x, u, t)
    k2 = ode.f(x+0.5*h*k1, u, t+0.5*h)
    dx = k2
    x_next = x + h*k2
    return x_next, dx

def rk2heun(x, h, u, t, ode):
    return x_next, dx

def rk3(x, h, u, t, ode):
    k1 = ode.f(x, u, t)
    k2 = ode.f(x+0.5*h*k1, u, t+0.5*h)
    k3 = ode.f(x-1*h*k1+2*h*k2, u, t+1*h)
    dx = (1/6)*k1 + (2/3)*k2 + (1/6)*k3
    x_next = x + (1/6)*h*k1 + (2/3)*h*k2 + (1/6)*h*k3
    return x_next, dx

def rk4(x, h, u, t, ode, jacobian=False):
    if(not jacobian):
        return x_next, dx
 
    return x_next, dx, phi_x, phi_u

def semi_implicit_euler(x, h, u, t, ode):
    return x_next, dx

def implicit_euler(x, h, u, t, ode):
    return x_next, dx