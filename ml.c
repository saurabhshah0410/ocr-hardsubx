#include<stdio.h>
#include<stdlib.h>
#include<string.h>

float predictTrees(DTreesImplForBoost* impl, const Point range, const Mat sample, int flags)
{
	int predictType = flags & PREDICT_MASK;
	int nvars = vector_size(impl->varIdx);
    if(nvars == 0)
    	nvars = vector_size(impl->varIdx);
    int i, ncats = (int)vector_size(impl->catOfs), nclasses = vector_size(impl->classLabels);
    int catbufsize = ncats > 0 ? nvars : 0;
    AutoBuffer buf = init_AutoBuffer(nclasses + catbufsize + 1);
    int* votes = buf.ptr;
    int* catbuf = votes + nclasses;
    const int* cvidx = (int*)vector_get(impl->compVarIdx, 0);
    const unsigned char* vtype = (unsigned char*)vector_get(varType, 0);
    const Point* cofs = (Point*)vector_get(catOfs, 0);
    const int* cmap = 0;
    const float* psample = (float*)ptr(sample, 0);
    const float* missingSubstPtr = 0;
    size_t sstep = 1;
    double sum = 0.;
    int lastClassIdx = -1;
    const float MISSED_VAL = FLT_MAX;

    for(i = 0; i < catbufsize; i++)
        catbuf[i] = -1;

    for(int ridx = range.x; ridx < range.y; ridx++)
    {
    	int nidx = *(int*)vector_get(impl->roots, ridx), prev = nidx, c = 0;

    	for(;;)
    	{
    		prev = nidx;
    		const Node node = (Node*)vector_get(impl->nodes, nidx);
    		if(node.split < 0)
                break;
			const Split split = (Split*)vector_get(impl->splits, node.split);
			int vi = split.varIdx;
            int ci = cvidx ? cvidx[vi] : vi;
            float val = psample[ci*sstep];

            if(val == MISSED_VAL)
            {
                if(!missingSubstPtr)
                {
                    nidx = node.defaultDir < 0 ? node.left : node.right;
                    continue;
                }
                val = missingSubstPtr[vi];
            }

            if(vtype[vi] == 0)
                nidx = val <= split.c ? node.left : node.right;
            else
            {
            	c = catbuf[ci];
                if(c < 0)
                {
                    int a = c = cofs[vi].x;
                    int b = cofs[vi].y;

                    int ival = round(val);

                    while(a < b)
                    {
                        c = (a + b) >> 1;
                        if(ival < cmap[c])
                            b = c;
                        else if(ival > cmap[c])
                            a = c+1;
                        else
                            break;
                    }

                    c -= cofs[vi].x;
                    catbuf[ci] = c;
                }
                const int* subset = (int*)vector_get(subsets, split.subsetOfs);
                unsigned u = c;
                nidx = CV_DTREE_CAT_DIR(u, subset) < 0 ? node.left : node.right;
            }
    	}
		sum += nodes[prev].value;
    }

    return (float)sum;
}

//Creates the empty model
static Boost* createBoost()
{
    Boost* obj = malloc(sizeof(Boost));
    obj->impl._isClassifier = false;
    obj->impl.params = init_TreeParams();
    obj->impl.params.CVFolds = 0;
    obj->impl.params.maxDepth = 1;
    obj->impl.bparams.boostType = 1;
    obj->impl.bparams.weakCount = 100;
    obj->impl.bparams.weightTrimRate = 0.95;
    vector_init(obj->impl.varIdx);
    vector_init(obj->impl.compVarIdx);
    vector_init(obj->impl.varType);
    vector_init(obj->impl.catOfs);
    vector_init(obj->impl.catMap);
    vector_init(obj->impl.roots);
    vector_init(obj->impl.nodes);
    vector_init(obj->impl.splits);
    vector_init(obj->impl.subsets);
    vector_init(obj->impl.classLabels);
    vector_init(obj->impl.missingSubst);
    vector_init(obj->impl.varMapping);
    vector_init(obj->impl.sumResult);
    obj->impl.w = malloc(sizeof(WorkData));
    return obj;
}

void read_ml(Boost* obj, const FileNode fn)
{
    FileNode _fn = FileNode_op(fn, "ntrees");
    int ntrees = int_op(_fn);
    readParams(&(obj->impl), fn);

    FileNode trees_node = FileNode_op(fn, "trees");
    FileNodeIterator it = begin(trees_node);

    for(int treeidx = 0; treeidx < ntrees; treeidx++, getNextIterator_(&it))
    {
        FileNode nfn = FileNode_op(fileNode(it.fs, (const CvFileNode*)(const void*)it.reader.ptr), "nodes");
        readTree(&(obj->impl), nfn);
    }
}


/** @brief Loads algorithm from the file

     @param filename Name of the file to read.
     @param objname The optional name of the node to read (if empty, the first top-level node will be used)

     This is static template method of Algorithm. It's usage is following (in the case of SVM):
     @code
     Ptr<SVM> svm = Algorithm::load<SVM>("my_svm_model.xml");
     @endcode
     In order to make this method work, the derived class must overwrite Algorithm::read(const
     FileNode& fn).
     */
Boost* load_ml(char* filename)
{
    FileStorage fs = fileStorage(filename, READ);
    FileNode fn = getFirstTopLevelNode(fs);
    Boost* obj = createBoost();
    read_ml(obj, fn);
    return obj;
}

//Predicts response(s) for the provided sample(s) 
void predict_ml(DTreesImplForBoost* impl, Mat samples, int flags)
{
	int rtype = 5;
	int i, nsamples = samples.rows;
	float retval = 1.0f;
	bool iscls = true;
	float scale = 1.f;

	nsamples = min(nsamples, 1);
	retval = predictTrees(init_Point(0, vector_size(impl->roots)), row_op(samples, init_Point(y, y+1)), flags)*scale;
	return retval;
}