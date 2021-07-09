#include "mex.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#define FEATURES_ARG     prhs[0]
#define K_ARG            prhs[1]
#define NUM_LABELED_ARG  prhs[2]  //assume features are arranged as such: [labeled; unlabeled]
#define PRINT_ITER_ARG  prhs[3]

#define NEIGHBORS_ARG plhs[0]
#define WEIGHTS_ARG   plhs[1]

/* types and comparator for sorting similarities while retaining indices */
typedef struct {
  unsigned int ind;
  double value;
} tagged_similarity;

int tagged_similarity_comparator(const void * x, const void * y) {
  double x_value = ((tagged_similarity *) x)->value;
  double y_value = ((tagged_similarity *) y)->value;
  
  if (x_value == y_value)
    return 0;
  
  return (x_value < y_value) ? 1 : -1;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  
  if ((nlhs != 2) || (nrhs != 4)) {
    mexErrMsgIdAndTxt("jaccard_nn:incorrect_arguments",
            "wrong number of arguments!");
  }
  
  mwSize num_points, num_features;
  mwIndex *ir, *jc;
  double *neighbors, *weights;
  
  unsigned int i, j, k, num_neighbors, num_labeled, print_iter, dot_product, *nnzs;
  unsigned char *feature_vector;
  tagged_similarity *similarities;
  
  num_points   = mxGetN(FEATURES_ARG);
  num_features = mxGetM(FEATURES_ARG);
  
  ir = mxGetIr(FEATURES_ARG);
  jc = mxGetJc(FEATURES_ARG);
  
  num_neighbors = (unsigned int) mxGetPr(K_ARG)[0];
  num_labeled = (unsigned int) mxGetPr(NUM_LABELED_ARG)[0];
  print_iter = (unsigned int) mxGetPr(PRINT_ITER_ARG)[0];
  assert(num_neighbors <= num_labeled);
  
  printf("initializing output variables ...\n");
  NEIGHBORS_ARG = mxCreateDoubleMatrix(num_points, num_neighbors, mxREAL);
  WEIGHTS_ARG   = mxCreateDoubleMatrix(num_points, num_neighbors, mxREAL);
  
  neighbors = mxGetPr(NEIGHBORS_ARG);
  weights   = mxGetPr(WEIGHTS_ARG);
  
  printf("initializing variables ...\n");
  nnzs              =     (unsigned  int *) mxMalloc(num_points   * sizeof(unsigned int));
  feature_vector    =     (unsigned char *) mxMalloc(num_features * sizeof(unsigned char));
  similarities      = (tagged_similarity *) mxMalloc(num_labeled  * sizeof(tagged_similarity));

  printf("getting number of nonzero features for each point...\n");
  for (i = 0; i < num_points; i++)
    nnzs[i] = jc[i + 1] - jc[i];
  
  printf("initializing similarities ...\n");
  for (j = 0; j < num_labeled; j++) {
    similarities[j].ind = j;
    similarities[j].value = -DBL_MAX;
  }
  
  for (i = 0; i < num_points; i++) {
    if (i % print_iter == 0) {
      printf("%d / %d ...\n", i, num_points);
    } 
    if (i >= num_labeled)  // only compute for unlabeled points
      memset(feature_vector, 0, num_features);

      for (j = jc[i]; j < jc[i + 1]; j++)
        feature_vector[ir[j]]++;

      for (j = 0; j < num_labeled; j++) {
        similarities[j].ind = j;

        dot_product = 0;
        for (k = jc[j]; k < jc[j + 1]; k++) {
          if (feature_vector[ir[k]]) {
            dot_product++;
          }
        }

        similarities[j].value = (double)(dot_product) / (nnzs[i] + nnzs[j] - dot_product);
      }

      qsort(similarities, num_labeled, sizeof(tagged_similarity), &tagged_similarity_comparator);
    
    for (k = 0; k < num_neighbors; k++) {
      neighbors[i + k * num_points] = similarities[k].ind + 1;
      weights[  i + k * num_points] = similarities[k].value;
    }
  }
}
