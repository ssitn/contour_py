# -*- coding: utf-8 -*-
"""
Created on Wed May 16 11:21:00 2019

@author: Sergey Sitnikov, s.l.sitnikov@gmail.com
"""

# This is a set of functions which are used for Elliptical Fourier Analysis of a closed contours. Please refer to each separate functions for details.
# points_connect() - a function used to connect 2 points or array of points with a straight line. It creates an array of pixel coordinates which 
#                    form a straight line between pair of input points.
#  
# points_arrange() - a function which will orders an array of coordinates of the contour so that each pixel is followed by the neares next pixel. 
# curvature2D() - calculates curvature of the contour in each point.
# centre_mass2D() - calculates the coordinates of object's center of mass.
# point_distance() - calculates distance between points in two input arrays. If input_array_2 is a single point, the function calculates distance between
#                    this point and points in input_array_1
# EllFT_coef()  - performs elliptical fourier transformation of the closed contours, calculates ELT coefficients.
# iEllFT_coef() - performs inversed elliptical fourier transformation, calculates coordinates of reconstructed contour.
#  


import numpy
import scipy
import math



#%% Connects points in 2D space with straight line (fills the gap between coordinates of points in the input array)
def points_connect(input_array, *args):
  contour = 0  
  unique_var = 0
  
  for vals in args:
    if vals == 'contour':
      contour = 1  # will connect last two points of the input array as well, making complete closed contour
    elif vals == 'unique':
      unique_var = 1 # will remove repeating points
      
  if sum(numpy.shape(list(input_array))) <= 3:  # tests if input array has at least 2 points to connect
      print('Cannot connect a single point')
      return input_array
  
  else:  
    perim_line = input_array
    perim_full = list() 
    
    for ii in range(len(perim_line)-1):                                         #calculates points of the connecting straight line
        x1 = perim_line[ii][0]
        y1 = perim_line[ii][1]
        x2 = perim_line[ii+1][0]
        y2 = perim_line[ii+1][1]
        
        if abs(x2-x1) < abs(y2-y1):
            y_cor = numpy.array(range(min(y1,y2),max(y1,y2)+1))
            if y2 < y1:
                y_cor = y_cor[::-1]
            x_cor_pres = y_cor*(x2-x1)/(y2-y1) + x1 - y1*(x2-x1)/(y2-y1)
            x_cor = numpy.round(x_cor_pres)
            x_cor = x_cor.astype(int)
        
        else:
            x_cor = numpy.array(range(min(x1,x2),max(x1,x2)+1))
            if x2 < x1:
                x_cor = x_cor[::-1]
            y_cor_pres = x_cor*(y2-y1)/(x2-x1) + y1 - x1*(y2-y1)/(x2-x1)
            y_cor = numpy.round(y_cor_pres)
            y_cor = y_cor.astype(int)
            
        for kk in range(len(x_cor)):
            perim_full.append((x_cor[kk],y_cor[kk]))
            
    if contour:                                                                 # connects the first and tje last points of the contour 
        ii = ii+1
        x1 = perim_line[ii][0]
        y1 = perim_line[ii][1]
        x2 = perim_line[0][0]
        y2 = perim_line[0][1]
        
        if abs(x2-x1) < abs(y2-y1):
            y_cor = numpy.array(range(min(y1,y2),max(y1,y2)+1))
            if y2 < y1:
                y_cor = y_cor[::-1]
            x_cor_pres = y_cor*(x2-x1)/(y2-y1) + x1 - y1*(x2-x1)/(y2-y1)
            x_cor = numpy.round(x_cor_pres)
            x_cor = x_cor.astype(int)
                
        else:
            x_cor = numpy.array(range(min(x1,x2),max(x1,x2)+1))
            if x2 < x1:
                x_cor = x_cor[::-1]
            y_cor_pres = x_cor*(y2-y1)/(x2-x1) + y1 - x1*(y2-y1)/(x2-x1)
            y_cor = numpy.round(y_cor_pres)
            y_cor = y_cor.astype(int)
            
        for kk in range(len(x_cor)):
            perim_full.append((x_cor[kk],y_cor[kk]))
    if unique_var:        
        p_unique = []
        for pair in perim_full:
            if pair not in p_unique:
                p_unique.append(pair)
    else:
        p_unique = perim_full            
    return p_unique
#%% rearanges points in input array so they every next point is the closest one to the previous
def points_arrange(input_array, *args):
    if len(args) > 0:
        start_ind = list(args)[0]
    else:
        start_ind = 0
    if start_ind > len(input_array)-1:
        print('Start point exceeds array length. Maximal starting index is ', len(input_array)-1)
        return []
    perim_idx_temp = input_array[:]
   
    perim_line = list()
    start = perim_idx_temp[start_ind]
  
    perim_line.append(start)
    start1 = start
    perim_idx_temp.remove(start)
    while len(perim_idx_temp)>0:
        min_dist = 500
        
        for ii in perim_idx_temp:
            distance = ((ii[0]-start[0])**2+(ii[1]-start[1])**2)**(1/2)
            if distance < min_dist:
              start1 = ii
              min_dist = distance
        start = start1
        perim_line.append(start)
        
        perim_idx_temp.remove(start)    

    return perim_line

#%% calculates curvature of the array of points (x,y)
def curvature2D(input_array):
    
    if numpy.shape(input_array)[1] != 2:
        print('Input array must be n by 2 size (X and Y Cartesian coordinates)')
    else:
        a = numpy.array(input_array)
        dx_dt = numpy.gradient(a[:, 0])
        dy_dt = numpy.gradient(a[:, 1])
        velocity = numpy.array([ [dx_dt[i], dy_dt[i]] for i in range(dx_dt.size)])
        ds_dt = numpy.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
        tangent = numpy.array([1/ds_dt] * 2).transpose() * velocity
        tangent_x = tangent[:, 0]
        tangent_y = tangent[:, 1]
        deriv_tangent_x = numpy.gradient(tangent_x)
        deriv_tangent_y = numpy.gradient(tangent_y)
        dT_dt = numpy.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
        length_dT_dt = numpy.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
        normal = numpy.array([1/length_dT_dt] * 2).transpose() * dT_dt
        d2s_dt2 = numpy.gradient(ds_dt)
        d2x_dt2 = numpy.gradient(dx_dt)
        d2y_dt2 = numpy.gradient(dy_dt)
        
        curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
        curvature = numpy.array(curvature)
        t_component = numpy.array([d2s_dt2] * 2).transpose()
        n_component = numpy.array([curvature * ds_dt * ds_dt] * 2).transpose()
        
        acceleration = t_component * tangent + n_component * normal
    
    return curvature, t_component, n_component, acceleration

#%% mean filter
    
def smooth_mean(y, box_pts):
    box = numpy.ones(box_pts)/box_pts
    y_smooth = numpy.convolve(y, box, mode='same')
    return y_smooth


#%% calculates center of mass of the 2D object
    
def centre_mass2D(input_array_xy, *args, **kwargs):
    data_type = numpy.float
    input_array = numpy.array(input_array_xy)
    for key, vals in kwargs.items():
        if key == 'dtype':
            data_type = vals  
    if len(args) > 0:
        mass_array = list(args)[0]
        print(sum(mass_array))
    else:
        mass_array = numpy.ones(numpy.shape(input_array)[0]) 
    center_x = sum(input_array[:,0]*mass_array)/sum(mass_array)
    center_y = sum(input_array[:,1]*mass_array)/sum(mass_array)
    if data_type == 'round':
        center_xy = numpy.array([numpy.around(center_x),numpy.around(center_y)], dtype = numpy.int)
    else:
        center_xy = numpy.array([center_x, center_y], dtype = data_type)
    total_mass = sum(mass_array)
    
    return center_xy, total_mass

#%% calculates distance between points in two arrays (x1,y1) <-> (x2,y2). Input array 2 can be a single point (x,y)
    
def point_distance(input_array1, input_array2):
        
    input_array1_xy = numpy.array(input_array1)
    
    if numpy.size(input_array2)>2:
        input_array2_xy = numpy.array(input_array2)
    elif numpy.size(input_array2) == 2:
        input_array2_xy = input_array2*numpy.ones(numpy.shape(input_array1))
    else:
        print('coordinates of second point set are incomplete')
        return
    
    point_dist_array = []
    
    for ii in range(numpy.shape(input_array2_xy)[0]):
        point_dist_array.append(numpy.sqrt((input_array1_xy[ii,0]-input_array2_xy[ii,0])**2+(input_array1_xy[ii,1]-input_array2_xy[ii,1])**2))
    point_dist_array = numpy.array(point_dist_array)
    return point_dist_array

#%% Calculates ellyptic fourier transformation coefficients

def EllFT_coef(input_array, N,*args):
    loco = 0
    full_coef = 0
    for vals in args:
        if vals == 'loco':
            loco = 1 
        elif vals == 'full':
            full_coef = 1
    dt = []
    t = []
    perim_full = input_array[:]
    for ii in range(len(perim_full)):
        dt.append(numpy.sqrt((perim_full[ii][0]-perim_full[ii-1][0])**2+(perim_full[ii][1]-perim_full[ii-1][1])**2))
        t.append(sum(dt)) 
    
    T = t[-1]
    alph0 = 0
    gamm0 = 0
    if full_coef:
        for ii in range(len(t)):
            if ii == 0:
                ksi = 0
                eps = 0
            elif ii==1:
                jj = 0
                ksi = ksi + perim_full[jj][0]-perim_full[jj-1][0] + (perim_full[ii][0]-perim_full[ii-1][0])/dt[ii]*dt[jj] 
                eps = eps + perim_full[jj][1]-perim_full[jj-1][1] + (perim_full[ii][1]-perim_full[ii-1][1])/dt[ii]*dt[jj]
            else:
                eps = 0
                ksi = 0
                for jj in range(ii-1):
                    ksi = ksi + perim_full[jj][0]-perim_full[jj-1][0] + (perim_full[ii][0]-perim_full[ii-1][0])/dt[ii]*dt[jj] 
                    eps = eps + perim_full[jj][1]-perim_full[jj-1][1] + (perim_full[ii][1]-perim_full[ii-1][1])/dt[ii]*dt[jj]
            if ii == 0:
                alph0 = alph0 + 0.5*(perim_full[ii][0]-perim_full[ii-1][0])/dt[ii]*(t[ii]**2)
                gamm0 = gamm0 + 0.5*(perim_full[ii][1]-perim_full[ii-1][1])/dt[ii]*(t[ii]**2) 
            else:
                alph0 = alph0 + 0.5*(perim_full[ii][0]-perim_full[ii-1][0])/dt[ii]*(t[ii]**2-t[ii-1]**2) + ksi*dt[ii] 
                gamm0 = gamm0 + 0.5*(perim_full[ii][1]-perim_full[ii-1][1])/dt[ii]*(t[ii]**2-t[ii-1]**2) + eps*dt[ii] 
        alph0 = -alph0/T + perim_full[-1][0]
        gamm0 = -gamm0/T + perim_full[-1][1]
    A0 = numpy.array([alph0,gamm0], dtype = numpy.float)
    A_out      = []
    A_tr_out   = []
    lam_p_out  = []
    lam_n_out  = []
    zeta_p_out = []
    zeta_n_out = []
    L_out      = []
    for n in range(1,N+1):
        alph = 0
        bet  = 0
        gamm = 0
        delt = 0
        for ii in range(len(t)):
            alph = alph + (perim_full[ii][0]-perim_full[ii-1][0])/dt[ii]*(math.cos(2*n*math.pi*t[ii]/T)-math.cos(2*n*math.pi*t[ii-1]/T))
            bet  = bet  + (perim_full[ii][0]-perim_full[ii-1][0])/dt[ii]*(math.sin(2*n*math.pi*t[ii]/T)-math.sin(2*n*math.pi*t[ii-1]/T))
            gamm = gamm + (perim_full[ii][1]-perim_full[ii-1][1])/dt[ii]*(math.cos(2*n*math.pi*t[ii]/T)-math.cos(2*n*math.pi*t[ii-1]/T))
            delt = delt + (perim_full[ii][1]-perim_full[ii-1][1])/dt[ii]*(math.sin(2*n*math.pi*t[ii]/T)-math.sin(2*n*math.pi*t[ii-1]/T))
            
        alpha_n = T/(2*n**2*math.pi**2)*alph
        beta_n  = T/(2*n**2*math.pi**2)*bet
        gamma_n = T/(2*n**2*math.pi**2)*gamm
        delta_n = T/(2*n**2*math.pi**2)*delt
        
        A = numpy.array([[alpha_n, beta_n],[gamma_n, delta_n]])
        A_out.append(A)
        if loco:
            if n==1:
                r = alpha_n*delta_n - beta_n*gamma_n
                
                tau1 = 0.5*math.atan2(2*(alpha_n*beta_n + gamma_n*delta_n),alpha_n**2 + gamma_n**2 - beta_n**2 - delta_n**2)
                A_prime = numpy.matmul(A, [[math.cos(tau1), -math.sin(tau1)],[math.sin(tau1), math.cos(tau1)]])
                
                tau_prime = math.atan2(A_prime[1,0],A_prime[0,0])
                
                if numpy.sign(tau_prime) == -1:
                    tau = tau1 + math.pi
                else:
                    tau = tau1
            
            A_tr = numpy.matmul(A, [[math.cos(n*tau), -math.sin(n*tau)],[math.sin(n*tau), math.cos(n*tau)]])
            
            if numpy.sign(r) == -1:
                A_tr[0,1] = -A_tr[0,1]
                A_tr[1,1] = -A_tr[1,1]
        
            A_tr_out.append(A_tr)
            phi = 0.5*math.atan2(2*(A_tr[0,0]*A_tr[0,1] + A_tr[1,0]*A_tr[1,1]), A_tr[0,0]**2 + A_tr[1,0]**2 - A_tr[0,1]**2 - A_tr[1,1]**2)
            A_tr_temp = numpy.matmul(A_tr, [[math.cos(phi), -math.sin(phi)],[math.sin(phi), math.cos(phi)]])
            theta = math.atan2(A_tr_temp[1,0], A_tr_temp[0,0])
            Lam_mat = numpy.matmul(numpy.matmul([[math.cos(theta), math.sin(theta)],[-math.sin(theta), math.cos(theta)]],A_tr),[[math.cos(phi), -math.sin(phi)],[math.sin(phi), math.cos(phi)]])
            lam_p = (Lam_mat[0,0]+Lam_mat[1,1])/2
            lam_n = (Lam_mat[0,0]-Lam_mat[1,1])/2
            zeta_p = theta-phi
            zeta_n = -theta-phi
            if n==1:
                zeta1 = zeta_p
            L = numpy.sqrt(lam_p**2+lam_n**2+2*lam_p*lam_n*math.cos(zeta_p-zeta_n-2*zeta1))
            lam_p_out.append(lam_p)
            lam_n_out.append(lam_n)
            zeta_p_out.append(zeta_p)
            zeta_n_out.append(zeta_n)
            L_out.append(L)
        else:
            A_tr_out = A_out
    if full_coef:
        return A_tr_out, list([lam_p_out, lam_n_out, zeta_p_out, zeta_n_out]), L_out, A0
    else:
        return A_tr_out, list([lam_p_out, lam_n_out, zeta_p_out, zeta_n_out]), L_out


#%%
def iEllFT_coef(A0, An, N, init_xy, **kwargs):
    for key, vals in kwargs.items():
        if key == 'dtype':
            data_type = vals  
    dt = []
    t = []
    perim_full = init_xy
    for ii in range(len(perim_full)):
        dt.append(numpy.sqrt((perim_full[ii][0]-perim_full[ii-1][0])**2+(perim_full[ii][1]-perim_full[ii-1][1])**2))
        t.append(sum(dt)) 
    
    T = t[-1]
    xy_out = []
    
    for ii in range(len(t)):
        x = 0
        y = 0
        for jj in range(N):
            x = x + An[jj][0,0]*math.cos(2*(jj+1)*math.pi*t[ii]/T) + An[jj][0,1]*math.sin(2*(jj+1)*math.pi*t[ii]/T)
            y = y + An[jj][1,0]*math.cos(2*(jj+1)*math.pi*t[ii]/T) + An[jj][1,1]*math.sin(2*(jj+1)*math.pi*t[ii]/T)
        x = x + A0[0]
        y = y + A0[1]
        if 'data_type' in locals(): 
            if data_type == 'round_int':
                xy = (numpy.int(numpy.around(x)),numpy.int(numpy.around(y)))
            else:
                xy = (x, y)
        else:
            xy = (x, y)
        xy_out.append(xy)
        
    return xy_out

#%%
def curv_test(input_array):
    a = numpy.array(input_array)
    dx_dt = [0, numpy.diff(a[:, 0])]    
    dy_dt = [0,numpy.diff(a[:, 1])]
    d2x_dt2 = [0, numpy.diff(dx_dt)]    
    d2y_dt2 = [0,numpy.diff(dy_dt)]
    curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    
    return curvature
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print(' ')
