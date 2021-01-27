# PLoting libary

import matplotlib.pyplot as plt
import numpy as np

def plot_grid(X,P,title,name,SAVEFIG,PLOTFIG):

	Nx=len(X[:,0])-2
	Nsig=len(X[0,:])-2

	fig = plt.figure()
	ax = fig.add_subplot(111)

	coord = np.zeros((Nx+2,2,Nsig+2))
	temp_coord = np.zeros((2,Nsig+2))

	for i in range(Nx+2):
		for j in range(Nsig+2):
			coord[i,0,j] = X[i,j]
			coord[i,1,j] = P[i,j]

	for i in range(Nx+2):
		temp_coord = coord[i,:,:]
		temp = zip(*temp_coord)
		xs,ys = zip(*temp)
		ax.plot(xs, ys,"k")



	for i in range(Nx+2):
		for j in range(Nsig+2):
			coord[i,0,j] = X[j,i]
			coord[i,1,j] = P[j,i]


	for i in range(Nx+2):
		temp_coord = coord[i,:,:]
		temp = zip(*temp_coord)
		xs,ys = zip(*temp)
		ax.plot(xs, ys,"k")


	ax.set_xlim(-10, 110)
	ax.set_ylim(-2, 12)
	ax.set_title('Example of grid')
	ax.set_xlabel('x-axis')
	ax.set_ylabel('sigma-axis')

	if (SAVEFIG==1):
		plt.savefig(name)

	if (PLOTFIG!=0):
		plt.ion()
		plt.draw()


def plot_contour(fig,T,t,nt,X,P,colors,title,name,SAVEFIG,PLOTFIG):
	#plt.figure(fig,figsize=(10,5))
	plt.figure(fig)
	plt.clf()
	plt.gca().invert_yaxis()


	#plt.axis("equal")
	if len(colors)==0:
		plt.contourf(X,P,T)
	else:
		plt.contourf(X,P,T,colors)
	plt.colorbar(extend="both",format="%.2e")
	#plt.ylim(pB,p0)
	NewTitle = title + " at t=" + str(t)
	plt.title(NewTitle,fontsize=12)


	# To set y axis
	# plt.ylim(-2,2)
	# See http://matplotlib.org/api/pyplot_api.html



	#plt.colorbar.set_clim(vmin=0,vmax=2.0)
	if (SAVEFIG==1):
		if (nt<10):
			filename = name + '_000000000' + str(nt) + '.png'
		elif (nt<100):
			filename = name + '_00000000' + str(nt) + '.png'
		elif (nt<1000):
			filename = name + '_0000000' + str(nt) + '.png'
		elif (nt<10000):
			filename = name + '_000000' + str(nt) + '.png'
		elif (nt<100000):
			filename = name + '_00000' + str(nt) + '.png'
		elif (nt<1000000):
			filename = name + '_0000' + str(nt) + '.png'
		elif (nt<10000000):
			filename = name + '_000' + str(nt) + '.png'
		elif (nt<100000000):
			filename = name + '_00' + str(nt) + '.png'
		elif (nt<1000000000):
			filename = name + '_0' + str(nt) + '.png'
		else:
			filename = name + '_' + str(nt) + '.png'
		plt.savefig(filename)
	if (PLOTFIG!=0):
		plt.ion()
		plt.draw()
