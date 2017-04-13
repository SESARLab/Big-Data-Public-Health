from matplotlib import pyplot as plt
import numpy as np
import sys, getopt, csv

def main(argv):
   inputfile = ''
   fpr=np.array([])
   tpr=np.array([])
   roc = 0.0
   try:
      opts, args = getopt.getopt(argv,"hi:r:",["ifile=","roc="])
   except getopt.GetoptError:
      print 'plot.py -i <inputfile> -r <rocvalue>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'plot.py -i <inputfile> -r <rocvalue>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
         inputfile = "C:\\Users\\geso8_000\\Documents\\esempio-spark\\ROC\\roc.csv"
      elif opt in ("-r", "--roc"):
         roc = float(arg)
   
   print 'Input file is "', inputfile
   print 'Output file is "', roc

   with open(inputfile, 'rb') as csvfile:
   		spamreader = csv.reader(csvfile, delimiter=',')
   		for row in spamreader:
   			fpr = np.append(fpr, [row[0]])
   			tpr = np.append(tpr, [row[1]])
   			print "CSV loaded"
   			
   plt.figure()
   lw = 2
   plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc)
   plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
   plt.xlim([0.0, 1.0])
   plt.ylim([0.0, 1.05])
   plt.xlabel('False Positive Rate')
   plt.ylabel('True Positive Rate')
   plt.title('Receiver operating characteristic')
   plt.legend(loc="lower right")
   plt.show()


if __name__ == "__main__":
   main(sys.argv[1:])

