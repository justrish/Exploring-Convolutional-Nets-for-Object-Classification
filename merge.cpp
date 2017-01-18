#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

using namespace std;

/** @function main */
int main( int argc, char** argv )
{
  ifstream ftrain0("//mnt//share//ILSVRC2015//train0.txt");
  ifstream ftrain1("//mnt//share//ILSVRC2015//train1.txt");
  ifstream ftrain2("//mnt//share//ILSVRC2015//train2.txt");
  ofstream fsave;
  fsave.open("//mnt//share//ILSVRC2015//train_pca_alex.txt", ios::out | ios::app );
  
  if (ftrain0.is_open() && ftrain1.is_open() && ftrain2.is_open() )
  {
	  string line0;
	  string line1;
	  string line2;
      while ( 1 )
      {
    	getline (ftrain0,line0);
		getline (ftrain1,line1);
		getline (ftrain2,line2);
		
		if (ftrain0.good()||ftrain0.eof() )
		  fsave << line0 << endl;
	    if (ftrain1.good()||ftrain1.eof() )
		  fsave << line1 << endl;
		if (ftrain2.good()||ftrain2.eof())
		  fsave << line2 << endl;
		if (ftrain0.good() || ftrain1.good() || ftrain2.good())	
        {
		}			
		else
			break;

	   }
	  
	  ftrain0.close();
	  ftrain1.close();
      ftrain2.close();
      fsave.close();

  }
  else cout << "Unable to open file";

  return 0;
 }