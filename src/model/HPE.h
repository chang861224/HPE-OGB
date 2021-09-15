#ifndef HPE_H
#define HPE_H

#include "../../smore/src/model/LINE.h"

/*****
 * HPE
 * **************************************************************/

class HPE: public LINE {

    public:
        
        HPE();
        ~HPE();

        void SaveWeights(string);

        // model function
        void Init(int, string);
        void Train(int, int, int, double, double, int);

};


#endif
