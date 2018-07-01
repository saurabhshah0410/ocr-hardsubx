#include <stdio.h>

#define CV_TREE_NODE_FIELDS(node_type)                                 \
    int       flags;             /**< Miscellaneous flags.     */      \
    int       header_size;       /**< Size of sequence header. */      \
    struct    node_type* h_prev; /**< Previous sequence.       */      \
    struct    node_type* h_next; /**< Next sequence.           */      \
    struct    node_type* v_prev; /**< 2nd previous sequence.   */      \
    struct    node_type* v_next  /**< 2nd next sequence.       */

#define CV_SEQUENCE_FIELDS()                                                         \
    CV_TREE_NODE_FIELDS(Seq);                                                        \
    int       total;           /**< Total number of elements.            */          \
    int       elem_size;       /**< Size of sequence element in bytes.   */          \
    signed char*    block_max; /**< Maximal bound of the last block.     */          \
    signed char*    ptr;       /**< Current write pointer.               */          \
    int       delta_elems;     /**< Grow seq this many at a time.        */          \
    MemStorage* storage;       /**< Where the seq is stored.             */          \
    SeqBlock* free_blocks;     /**< Free blocks list.                    */          \
    SeqBlock* first;           /**< Pointer to the first sequence block. */

#define CV_CONTOUR_FIELDS()  \
    CV_SEQUENCE_FIELDS()     \
    Rect rect;               \
    int color;               \
    int reserved[3];

#define CV_SET_FIELDS()      \
    CV_SEQUENCE_FIELDS()     \
    SetElem* free_elems;     \
    int active_count;

#define CV_SET_ELEM_FIELDS(elem_type)   \
    int  flags;                         \
    struct elem_type* next_free;

#define CV_IS_MASK_ARR(mat)             \
    (((mat)->type & (CV_MAT_TYPE_MASK & ~CV_8SC1)) == 0)

#define ICV_FREE_PTR(storage)  \
    ((signed char*)(storage)->top + (storage)->block_size - (storage)->free_space)

typedef struct Mat
{
	/*! includes several bit-fields:
	     - the magic signature
	     - continuity flag
	     - depth
	     - number of channels
	*/
	int flags;
	//! the number of rows and columns
	int rows, cols;
	//! pointer to the data
	unsigned char* data;

    //! helper fields used in locateROI and adjustROI
    const unsigned char* datastart;
    const unsigned char* dataend;
    const unsigned char* datalimit;

    size_t step[2];
} Mat ;

typedef struct  MatIterator
{
    //! the iterated arrays
    const Mat** arrays;
    //! the current planes
    Mat* planes;
    //! data pointers
    uchar** ptrs;
    //! the number of arrays
    int narrays;
    //! the number of hyper-planes that the iterator steps through
    size_t nplanes;
    //! the size of each segment (in elements)
    size_t size;

    int iterdepth;
    size_t idx;
} MatIterator ;

typedef struct RGBtoGray
{
    int srccn;
    int tab[256*3];
} RGBtoGray ;

RGBtoGray RGB2Gray(int _srccn, int blueIdx, const int* coeffs)
{
    RGBtoGray rg;
    rg.srccn = _srccn;
    const int coeffs0[] = {R2Y, G2Y, B2Y};
    if(!coeffs) 
        coeffs = coeffs0;

    int b = 0, g = 0, r = (1 << 13);
    int db = coeffs[blueIdx^2], dg = coeffs[1], dr = coeffs[blueIdx];

    for(int i = 0; i < 256; i++, b += db, g += dg, r += dr)
    {
        rg.tab[i] = b;
        rg.tab[i+256] = g;
        rg.tab[i+512] = r;
    }
    return rg;
}

typedef struct CvtHelper
{
    Mat src, dst;
    int depth, scn;
    Point dstSz;
} CvtHelper ;

typedef struct Point
{
    int x;
    int y;
} Point ;

// Represents a pair of ER's
typedef struct region_pair
{
    Point a;
    Point b;
} region_pair ;

typedef struct Rect
{
	int x;					  //!< x coordinate of the top-left corner
	int y;					  //!< y coordinate of the top-left corner 
	int width;				  //!< width of the rectangle
	int height;				  //!< height of the rectangle
} Rect ;

typedef struct Scalar
{
    double val[4];
} Scalar ;


/** @brief The ERStat structure represents a class-specific Extremal Region (ER).

An ER is a 4-connected set of pixels with all its grey-level values smaller than the values in its
outer boundary. A class-specific ER is selected (using a classifier) from all the ER's in the
component tree of the image. :
 */
typedef struct ERStat
{
	//! Incrementally Computable Descriptors
	int pixel;
	int level;
	int area;
    int perimeter;
    int euler;                 //!< Euler's number
    Rect rect;				   // Rect_<int>
    double raw_moments[2];     //!< order 1 raw moments to derive the centroid
	double central_moments[3]; //!< order 2 central moments to construct the covariance matrix
	vector* crossings;     	   //!< horizontal crossings
	float med_crossings;       //!< median of the crossings at three different height levels

	//! stage 2 features
	float hole_area_ratio;
	float convex_hull_ratio;
	float num_inflexion_points;

	//! pixel list after the 2nd stage
	int** pixels;

	//! probability that the ER belongs to the class we are looking for
	double probability;

	//! pointers preserving the tree structure of the component tree --> might be optional
	ERStat* parent;
	ERStat* child;
	ERStat* next;
	ERStat* prev;

	//! whenever the regions is a local maxima of the probability
	bool local_maxima;
	ERStat* max_probability_ancestor;
	ERStat* min_probability_ancestor;
} ERStat ;

// the struct implementing the interface for the 1st and 2nd stages of Neumann and Matas algorithm
typedef struct ERFilterNM
{
	float minProbability;   
	bool  nonMaxSuppression;
	float minProbabilityDiff;
	int thresholdDelta;
	float maxArea; 
    float minArea;

    ERClassifierNM* classifier;

    // count of the rejected/accepted regions
	int num_rejected_regions;
	int num_accepted_regions;

	//! Input/output regions
	vector* regions; //ERStat

	//! image mask used for feature calculations
	Mat region_mask;
} ERFilterNM ;

typedef struct ERClassifierNM
{
	Boost* boost;
} ERClassifierNM ;

typedef struct FFillSegment
{
    unsigned short y;
    unsigned short l;
    unsigned short r;
    unsigned short prevl;
    unsigned short prevr;
    short dir;
} FFillSegment ;

typedef struct ConnectedComp
{
    Rect rect;
    Point pt;
    int threshold;
    int label;
    int area;
    int harea;
    int carea;
    int perimeter;
    int nholes;
    int ninflections;
    double mx;
    double my;
    Scalar avg;
    Scalar sdv;
} ConnectedComp;

typedef struct CompHistory
{
    CompHistory* child_;
    CompHistory* parent_;
    CompHistory* next_;
    int val;
    int size;
    float var;
    int head;
    bool checked;
} CompHistory;

typedef struct Diff8uC1
{
    unsigned char lo, interval;
} Diff8uC1 ;

typedef struct Diff8uC3
{
    unsigned char lo[3], interval[3];
} Diff8uC3 ;

typedef struct ThresholdRunner
{
    Mat src;
    Mat dst;

    double thresh;
    double maxval;
    int thresholdType;
} ThresholdRunner;

typedef struct CvtColorLoop_Invoker
{
    const unsigned char* src_data;
    const size_t src_step;
    unsigned char * dst_data;
    const size_t dst_step;
    const int width;
    const RGBtoGray cvt;
} CvtColorLoop_Invoker ;

typedef struct TreeNode
{
    int       flags;            /* micsellaneous flags      */
    int       header_size;      /* size of sequence header  */
    struct    TreeNode* h_prev; /* previous sequence        */
    struct    TreeNode* h_next; /* next sequence            */
    struct    TreeNode* v_prev; /* 2nd previous sequence    */
    struct    TreeNode* v_next; /* 2nd next sequence        */
} TreeNode;


typedef struct Chain
{
    CV_SEQUENCE_FIELDS()
    Point  origin;
} Chain ;

typedef struct SetElem
{
    CV_SET_ELEM_FIELDS(SetElem)
} SetElem;

typedef struct Set
{
    CV_SET_FIELDS()
} Set;

typedef struct Contour
{
    CV_CONTOUR_FIELDS()
} Contour;

typedef struct Seq
{
    CV_SEQUENCE_FIELDS()
} Seq;

typedef struct SeqBlock
{
    struct SeqBlock*  prev;     /**< Previous sequence block.                   */
    struct SeqBlock*  next;     /**< Next sequence block.                       */
    int    start_index;         /**< Index of the first element in the block +  */
                                /**< sequence->first->start_index.              */
    int    count;               /**< Number of elements in the block.           */
    signed char* data;          /**< Pointer to the first element of the block. */
} SeqBlock ;

typedef struct MemBlock
{
    struct MemBlock*  prev;
    struct MemBlock*  next;
} MemBlock ;

typedef struct MemStorage
{
    int signature;
    MemBlock* bottom;            /**< First allocated block.                   */
    MemBlock* top;               /**< Current memory block - top of the stack. */
    struct  MemStorage* parent;  /**< We get new blocks from parent as needed. */
    int block_size;              /**< Block size.                              */
    int free_space;              /**< Remaining free space in current block.   */
} MemStorage ;

typedef struct MemStoragePos
{
    MemBlock* top;
    int free_space;
} MemStoragePos ;

typedef struct ContourInfo
{
    int flags;
    struct ContourInfo *next;       /* next contour with the same mark value            */
    struct ContourInfo *parent;     /* information about parent contour                 */
    Seq *contour;                   /* corresponding contour (may be 0, if rejected)    */
    Rect rect;                      /* bounding rectangle                               */
    Point origin;                   /* origin point (where the contour was traced from) */
    int is_hole;                    /* hole flag                                        */
} ContourInfo ;

typedef struct ContourScanner
{
    MemStorage *storage1;             /* contains fetched contours                              */
    MemStorage *storage2;             /* contains approximated contours
                                   (!=storage1 if approx_method2 != approx_method1)             */
    MemStorage *cinfo_storage;        /* contains _CvContourInfo nodes                          */
    Set *cinfo_set;                   /* set of _CvContourInfo nodes                            */
    MemStoragePos initial_pos;        /* starting storage pos                                   */
    MemStoragePos backup_pos;         /* beginning of the latest approx. contour                */
    MemStoragePos backup_pos2;        /* ending of the latest approx. contour                   */
    signed char *img0;                /* image origin                                           */
    signed char *img;                 /* current image row                                      */
    int img_step;                     /* image step                                             */
    Point img_size;                   /* ROI size                                               */
    Point offset;                     /* ROI offset: coordinates, added to each contour point   */
    Point pt;                         /* current scanner position                               */
    Point lnbd;                       /* position of the last met contour                       */
    int nbd;                          /* current mark val                                       */
    ContourInfo *l_cinfo;             /* information about latest approx. contour               */
    ContourInfo cinfo_temp;           /* temporary var which is used in simple modes            */
    ContourInfo frame_info;           /* information about frame                                */
    Seq frame;                        /* frame itself                                           */
    int approx_method1;               /* approx method when tracing                             */
    int approx_method2;               /* final approx method                                    */
    int mode;                         /* contour scanning mode:
                                           0 - external only
                                           1 - all the contours w/o any hierarchy
                                           2 - connected components (i.e. two-level structure -
                                           external contours and holes),
                                           3 - full hierarchy;
                                           4 - connected components of a multi-level image
                                      */
    int subst_flag;
    int seq_type1;                    /* type of fetched contours                               */
    int header_size1;                 /* hdr size of fetched contours                           */
    int elem_size1;                   /* elem size of fetched contours                          */
    int seq_type2;                    /*                                                        */
    int header_size2;                 /* the same for approx. contours                          */
    int elem_size2;                   /*                                                        */
    ContourInfo *cinfo_table[128];
} ContourScanner;

typedef struct AutoBuffer
{
    size_t fixed_size;
    //! pointer to the real buffer, can point to buf if the buffer is small enough
    int* ptr;
    //! size of the real buffer
    size_t sz;
    //! pre-allocated buffer. At least 1 element to confirm C++ standard requirements
    int buf[(fixed_size > 0) ? fixed_size : 1];
} AutoBuffer ;

AutoBuffer init_AutoBuffer(size_t _size)
{
    AutoBuffer ab;
    ab.fixed_size = 1024/sizeof(int)+8;
    ab.ptr = ab.buf;
    ab.sz = ab.fixed_size;
    allocateAB(&ab, _size);
    return ab;
}

void allocateAB(AutoBuffer* ab, size_t _size)
{
    if(_size <= ab->sz)
    {
        ab->sz = _size;
        return;
    }
    deallocateAB(ab);
    ab->sz = _size;
    if(_size > fixed_size)
    {
        ptr = new int[_size];
    }
}

void deallocateAB(AutoBuffer* ab)
{
    if(ab->ptr != ab->buf)
    {
        free(ab->ptr);
        ab->ptr = ab->buf;
        ab->sz = ab->fixed_size;
    }
}

typedef void (*BinaryFunc)(const unsigned char* src1, size_t step1,
                       const unsigned char* src2, size_t step2,
                       unsigned char* dst, size_t step, Point sz,
                       void*);

typedef void (*SplitFunc)(const unsigned char* src, unsigned char** dst, int len, int cn);

int updateContinuityFlag(Mat* m)
{
    int i, j;
    int sz[] = {m->rows, m->cols};
    
    for( i = 0; i < 2; i++ )
    {
        if(sz[i] > 1)
            break;
    }
    
    uint64_t t = (uint64_t)sz[i]*CV_MAT_CN(flags);
    for(j = 1; j > i; j--)
    {
        t *= size[j];
        if(m->step[j]*size[j] < m->step[j-1] )
            break;
    }

    if(j <= i && t == (uint64_t)(int)t)
        return flags | CONTINUOUS_FLAG;

    return flags & CONTINUOUS_FLAG;
}

size_t elemSize(Mat m)
{
    return m.step[1];
}

size_t elemSize1(Mat m)
{
    return CV_ELEM_SIZE1(flags);
}

bool isContinuous(Mat m)
{
    return (m.flags & CONTINUOUS_FLAG) != 0;
}

bool isSubmatrix(Mat m)
{
    return (m.flags & SUBMATRIX_FLAG) != 0;
}

void release(Mat* m)
{
    m->datastart = m->dataend = m->datalimit = m->data = 0;
    m->rows = m->cols = 0;
    free(m->step);
}

int type(Mat m)
{
    return CV_MAT_TYPE(m.flags);
}

inline bool empty(Mat m)
{
    return (m.data == 0);
}

int depth(Mat m)
{
    return (m.flags & CV_MAT_DEPTH_MASK);
    // #define CV_DEPTH_MAX  (1 << CV_CN_SHIFT) 
    // #define CV_MAT_DEPTH_MASK       (CV_DEPTH_MAX - 1)
}

int channels(Mat m)
{
    return CV_MAT_CN(m.flags);
}

unsigned char* ptr(Mat m, int i)
{
    return m.data + m.step[0] * i;
}

bool fixedType(Mat m)
{
    return (m.flags & FIXED_TYPE) == FIXED_TYPE;
}

Point getContinuousSize_(int flags, int cols, int rows, int widthScale)
{
    int64_t sz = (int64_t)cols * rows * widthScale;
    return (flags & CONTINUOUS_FLAG) != 0 &&
        (int)sz == sz ? init_Point((int)sz, 1) : init_Point(cols * widthScale, rows);
}

Point getContinuousSize(Mat m1, Mat m2, int widthScale)
{
    return getContinuousSize_(m1.flags & m2.flags,
                              m1.cols, m1.rows, widthScale);
}

void copyTo(Mat src, Mat* dst)
{
    int dtype = type(*dst);
    if(fixedType(*dst) && dtype != type(src))
    {
        convertTo(&src, dst, dtype, 1, 0);
        return;
    }

    if(empty(src))
    {
        release(dst);
        return;
    }

    create(dst, src.rows, src.cols, type(src));
    if(src.data == dst->data)
            return;

    if(src.rows > 0 && src.cols > 0)
    {
        const unsigned char* sptr = src.data;
        unsigned char* dptr = dst->data;

        Point sz = getContinuousSize(src, dst);
        size_t len = sz.x*elemSize(src);

        for(; sz.y-- ; sptr += src.step[0], dptr += dst.step[0])
                memcpy(dptr, sptr, len);
    }
    return;
}

void convertTo(Mat* src, Mat* dst, int _type, double alpha/*1*/, double beta/*0*/)
{
    if(empty(*src))
    {
        release(dst);
        return;
    }

    bool noScale = fabs(alpha-1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;
    if( _type < 0 )
        _type = fixedType(*dst) ? type(*dst) : type(*src);
    else
         _type = CV_MAKETYPE(CV_MAT_DEPTH(_type), channels(*src));

    int sdepth = depth(*src), ddepth = CV_MAT_DEPTH(_type);
    if(sdepth == ddepth && noScale)
    {
        copyTo(*src, dst);
        return;
    }

    create(dst, src->rows, src->cols, _type);
    BinaryFunc func = noScale ? getConvertFunc(sdepth, ddepth) : getConvertScaleFunc(sdepth, ddepth);

    double scale[] = {alpha, beta};
    int cn = channels(*src);

    Point sz = getContinuousSize(src->flags & dst->flags, src->cols, src->rows, cn);
    func(src->data, (src->step)[0], 0, 0, dst->data, (dst->step)[0], sz, scale);
}

static inline int Align(int size, int align)
{
    return (size+align-1) & -align;
}

static inline unsigned char** alignPtr(unsigned char** ptr, int n)
{
    return (unsigned char**)(((size_t)ptr + n-1) & -n);
}

int AlignLeft(int size, int align)
{
    return size & -align;
}

void scalarToRawData(const Scalar s, unsigned char* buf, int type, int unroll_to)
{
    int i;
    const int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    for(i = 0; i < cn;i++)
        buf[i] = s.val[i];
    for(; i < unroll_to; i++)
        buf[i] = buf[i-cn];
}

int min(int x1, int x2)
{
    return x1 > x2 ? x2 : x1;
}

int max(int x1, int x2)
{
    return x1 < x2 ? x2 : x1;
}

inline void create(Mat* m, int _rows, int _cols, int _type)
{
	_type &= TYPE_MASK;
    if(m->rows == _rows && m->cols == _cols && type(*m) == _type && m->data)
        return;

    int sz[] = {_rows, _cols};
    _type = MAT_TYPE(_type);
    // #define MAT_TYPE(flags)      ((flags) & MAT_TYPE_MASK)
    // #define MAT_TYPE_MASK        (CV_DEPTH_MAX*CV_CN_MAX - 1)

    m->rows = _rows;
    m->cols = _cols;
    m->flags = (_type & MAT_TYPE_MASK) | MAGIC_VAL;
    m->data = 0;

    size_t esz = CV_ELEM_SIZE(m->flags), esz1 = CV_ELEM_SIZE1(m->flags), total = esz;
    for(int i = 1; i >= 0; i--)
    {
        m->step[i] = total;
        int64_t total1 = (int64_t)total*sz[i];
        if((uint64_t)total1 != (size_t)total1)
            fatal("The total matrix size does not fit to \"size_t\" type");
        total = (size_t)total1;
    }
    if(m->rows * m->cols > 0)
    {
        total = total * m->rows * m->cols;
        unsigned char* udata = malloc(total + sizeof(void*) + CV_MALLOC_ALIGN);

        if(!udata)
            Fatal("Failed to allocate %llu bytes", (unsigned long long)size);

        unsigned char** adata = alignPtr((unsigned char**)udata + 1, CV_MALLOC_ALIGN);
        adata[-1] = udata;
        m->data = adata;
        assert(m->step[1] == (size_t)CV_ELEM_SIZE(flags));
    }
    m->flags = updateContinuityFlag(m);
    if(m->data)
    {
        m->datalimit = m->datastart + m->rows*m->step[0];
        if(m->rows > 0)
        {
            m->dataend = ptr(*m, 0) + m->cols*m->step[1];
            for(int i = 0; i < 1; i++)
                m->dataend += (sz[i] - 1)*m->step[i];
        }
        else
            m->dataend = m->datalimit;
    }
    else
        m->dataend = m->datalimit = 0;
}

Mat createusingRect(Mat _m, Rect roi)
{
    Mat m;
    m.flags = _m.flags;
    m.rows = roi.height;
    m.cols = roi.width;
    m.datastart = _m.datastart;
    m.datalimit = _m.datalimit;
    m.dataend = _m.dataend;
    m.data = _m.data + roi.y*_m.step[0];
    size_t esz = CV_ELEM_SIZE(m.flags);
    m.data += roi.x*esz;
    
    if(roi.width < _m.cols || roi.height < _m.rows)
        m.flags |= SUBMATRIX_FLAG;
    
    m.step[0] = _m.step[0]; m.step[1] = esz;
    updateContinuityFlag(&m);
    
    if(m.rows <= 0 || m.cols <= 0)
    {
        release(&m);
        m.rows = m.cols = 0;
    }

    return m;
}

// Sets all or some of the array elements to the specified value.
void createusingScalar(Mat* m, Scalar s)
{
    const Mat* arrays[] = {m};
    unsigned char* dptr;
    MatIterator it = matIterator(arrays, 0, &dptr, 1);
    size_t elsize = it.size*elemSize(*m);
    const int64_t* is = (const int64_t*)&s.val[0];

    if(is[0] == 0 && is[1] == 0 && is[2] == 0 && is[3] == 0)
    {
        for(size_t i = 0; i < it.nplanes; i++, getNextIterator(&it))
            memset(dptr, 0, elsize);
    }
    else
    {
        if(it.nplanes > 0)
        {
            double scalar[12];
            scalarToRawDoubleData(s, scalar, type(*m), 12);
            size_t blockSize = 12*elemSize1(*m);

            for(size_t j = 0; j < elsize; j += blockSize)
            {
                size_t sz = min(blockSize, elsize - j);
                memcpy(dptr+j, scalar, sz);
            }
        }
        for(size_t i = 1; i < it.nplanes; i++)
        {
            getNextIterator(&it);
            memcpy(dptr, m->data, elsize);
        }
    }
}

Mat zeros(int _rows, int _cols, int _type, void* _data, size_t _step) // _data = (void*)(size_t)0xEEEEEEEE
{
    Mat m;
    create(&m, _rows, _cols, _type);
    m.data = (unsigned char*)_data;
    m.datastart = (unsigned char*)_data;
    m.dataend = 0;
    m.datalimit = 0;

    size_t esz = CV_ELEM_SIZE(_type), esz1 = CV_ELEM_SIZE1(_type);
    size_t minstep = _cols*esz;


    if(_step == AUTO_STEP)
        _step = minstep;

    else
    {
        if (_step % esz1 != 0)
        {
            fatal("Step must be a multiple of esz1");
        }
    }

    m.step[0] = _step;
    m.step[1] = esz;
    m.datalimit = m.datastart + _step*m.rows;
    m.dataend = m.datalimit - _step + minstep;
    updateContinuityFlag(&m);
    return m;
}


Mat createusingRange(Mat m, int row0, row1)
{
    Mat src = m;
    if(row0 != 0 && row1 != src.rows)
    {
        assert(0 <= row0 && row0 <= row1
                       && row1 <= m.rows);
        m.rows = row1-row0;
        m.data += m.step*row0;
        m.flags |= SUBMATRIX_FLAG;
    }

    updateContinuityFlag(&m);

    if(m.rows <= 0 || m->cols <= 0)
    {
        release(&m);
        m.rows = m.cols = 0;
    }

    return m;
}

void locateROI(Mat src, Point* wholeSize, Point* ofs)
{
    assert((src.step)[0] > 0);
    size_t esz = elemSize(src), minstep;
    ptrdiff_t delta1 = src.data - src.datastart, delta2 = src.dataend - src.datastart;

    if(delta1 == 0)
        ofs->x = ofs->y = 0;

    else
    {
        ofs->y = (int)(delta1/step[0]);
        ofs->x = (int)((delta1 - step[0]*ofs->y)/esz);
    }
    minstep = (ofs->x + cols)*esz;
    wholeSize->y = (int)((delta2 - minstep)/step[0] + 1);
    wholeSize->y = max(wholeSize->y, ofs->y + src.rows);
    wholeSize->x = (int)((delta2 - step*(wholeSize->y-1))/esz);
    wholeSize->x = max(wholeSize->x, ofs->x + src.cols);
}

void adjustROI(Mat* src, int dtop, int dbottom, int dleft, int dright)
{
    assert((src->step)[0] > 0);
    Point *wholesize, *ofs;
    size_t esz = elemSize(*src);
    locateROI(*src, wholeSize, ofs);
    int row1 = min(max(ofs->y - dtop, 0), wholeSize->y), row2 = max(0, min(ofs->y + rows + dbottom, wholeSize->y));
    int col1 = min(max(ofs->x - dleft, 0), wholeSize->x), col2 = max(0, min(ofs->x + cols + dright, wholeSize->x));

    if(row1 > row2)
        swap(&row1, &row2);

    if(col1 > col2)
        swap(&col1, &col2);

    src->data += (row1 - ofs->y)*(m.step)[0] + (col1 - ofs->x)*esz;
    src->rows = row2 - row1; src->cols = col2 - col1;
    updateContinuityFlag(src);
}

void createVector(vector* v/* Mat */, int rows, int cols, int mtype, int i, bool allowTransposed, int fixedDepthMask)
{
    int sizebuf[2];
    mtype = CV_MAT_TYPE(mtype);

    if(i < 0)
    {
        size_t len = rows*cols > 0 ? rows + cols - 1 : 0, len0 = vector_size(v);

        vector_resize(v, len);
        return;   
    }

    assert(i < vector_size(v));
    Mat m = *((Mat*)vector_get(v, i));

    if(allowTransposed)
    {
        if(isContinuous(m))
            release((Mat*)vector_get(v, i));

        if(m.data && type(m) == mtype && m.rows == rows && m.cols == cols)
            return;
    }
    create((Mat*)vector_get(v, i), rows, cols, mtype);
    return;
}

void getNextIterator(MatIterator* it)
{
    if(it->idx >= it->nplanes-1)
        return;
    ++(it->idx);

    if(it->iterdepth == 1)
    {
        for(int i = 0; i < it->narrays; i++)
        {
            if(!it->ptrs[i])
                continue;
            it->ptrs[i] = it->arrays[i]->data + it->arrays[i]->step[0]->idx;
        }
    }
    else
    {
        for(int i = 0; i < it->narrays; i++)
        {
            Mat* A = arrays[i];
            if(!A->data)
                continue;
            int sz[2] = {A->rows, A->cols};
            int _idx = (int)it->idx;
            unsigned char* data = A->data;
            for(int j = it->iterdepth-1; j >= 0 && _idx > 0; j--)
            {
                int szi = sz[j], t = _idx/szi;
                data += (_idx - t * szi)*A->step[j];
                _idx = t;
            }
            it->ptrs[i] = data;
        }
    }
}

CvtColorLoop_Invoker cvtColorLoop_Invoker(const unsigned char* src_data_, size_t src_step_, unsigned char* dst_data_, size_t dst_step_, int width_, const RGBtoGray _cvt)
{
    CvtColorLoop_Invoker cli;
    cli.src_data = src_data_;
    cli.src_step = src_step_;
    cli.dst_data = dst_data_;
    cli.dst_step = dst_step_;
    cli.width = width_;
    cli.cvt = _cvt;
    return cli;
}

void Split(Mat src, Mat* mv)
{
    int k, depth = depth(src), cn = channels(src);
    if(cn == 1)
    {
        copyTo(src, &mv[0]);
        return;
    }

    for(k = 0;k < cn;k++)
        create(mv[k], src.rows, src.cols, depth);

    SplitFunc func = getSplitFunc(depth);
    size_t esz = elemSize(src), esz1 = elemSize1(src);
    size_t blocksize0 = (BLOCK_SIZE + esz-1)/esz;
    AutoBuffer _buf = init_AutoBuffer((cn+1)*(sizeof(Mat*) + sizeof(unsigned char*)) + 16);
    const Mat** arrays = (const Mat**)(unsigned char*)_buf;
    unsigned char** ptrs = (unsigned char**)alignPtr(arrays + cn + 1, 16);

    arrays[0] = &src;
    for(k = 0; k < cn; k++)
    {
        arrays[k+1] = &mv[k];
    }

    MatIterator it = matIterator(arrays, 0, ptrs, cn+1);
    size_t total = it.size;
    size_t blocksize = min((size_t)CV_SPLIT_MERGE_MAX_BLOCK_SIZE(cn), cn <= 4 ? total : min(total, blocksize0));

    for(size_t i = 0; i < it.nplanes; i++)
    {
        getNextIterator(&it);
        for(size_t j = 0; j < total; j += blocksize)
        {
            size_t bsz = min(total - j, blocksize);
            func(it.ptrs[0], &it.ptrs[1], (int)bsz, cn);

            if(j + blocksize < total)
            {
                ptrs[0] += bsz*esz;
                for(k = 0; k < cn; k++)
                    ptrs[k+1] += bsz*esz1;
            }
        }
    }
}

void split(Mat m, vector* dst/* Mat */)
{
    if(empty(m))
    {
        vector_free(dst);
        return;
    }

    int depth = depth(m);
    int cn = channels(m);

    createVector(dst, cn, 1, depth, -1, false, 0);
    for (int i = 0; i < cn; ++i)
        createVector(dst, m.rows, m.cols, depth, i);

    Split(m, vector_get(dst, 0));
}

int floor(double value)
{
    int i = (int)value;
    return i - (i > value);
}

int round(double value)
{
    /* it's ok if round does not comply with IEEE754 standard;
       the tests should allow +/-1 difference when the tested functions use round */
    return (int)(value + (value >= 0 ? 0.5 : -0.5));
}

Point init_Point(int _x, int _y)
{
    Point p;
    p.x = _x;
    p.y = _y;
    return p;
}

CvtHelper cvtHelper(Mat _src, Mat* _dst, int dcn)
{
    CvtHelper h;
    int stype = type(src);
    h.scn = CV_MAT_CN(stype), h.depth = CV_MAT_DEPTH(stype);

    if(_src == _dst)
        copyTo(_src, h.src);

    else
        h.src = _src;

    Point sz = init_Point(h.src.rows, h.src.cols);
    h.dstSz = sz;
    create(&h.dst, h.src.rows, h.src.cols, CV_MAKETYPE(depth, dcn));
    create(_dst, h.src.rows, h.src.cols, CV_MAKETYPE(depth, dcn));
    return h;
}

Scalar init_Scalar(double v1, double v2, double v3, double v4)
{
    Scalar s;
    s.val[0] = v1;
    s.val[1] = v2;
    s.val[2] = v3;
    s.val[3] = v4;
    return s;
}

region_pair init_region_pair(Point a, Point b)
{
    region_pair rp;
    rp.a = a;
    rp.b = b;
    return rp;
}
void init_ERStat(ERStat* erstat, int init_level, int init_pixel, int init_x, int init_y)
{
	erstat->level = init_level;
	erstat->pixel = init_pixel;
	erstat->area = 0;
	erstat->perimeter = 0;
	erstat->euler = 0;
	erstat->probability = 1.0;
	erstat->local_maxima = 0;
	erstat->parent = 0;
	erstat->child = 0;
	erstat->prev = 0;
	erstat->next = 0;
	erstat->max_probability_ancestor = 0;
	erstat->min_probability_ancestor = 0;
	erstat->rect = init_Rect(init_x, init_y, 1, 1);
	(erstat->raw_moments)[0] = 0.0;
	(erstat->raw_moments)[1] = 0.0;
	(erstat->central_moments)[0] = 0.0;
	(erstat->central_moments)[1] = 0.0;
	(erstat->central_moments)[2] = 0.0;
	erstat->crossings = malloc(sizeof(vector));
	vector_init(erstat->crossings);
	int val = 0;
	vector_add(erstat->crossings, &val);
}

Rect init_Rect(int _x, int _y, int _width, int _height)
{
    Rect r;
	r->x = _x;
	r->y = _y;
	r->width = _width;
	r->height = _height;
    return r;
}

Rect createRect(Point p, Point q)
{
    Rect r;
    r.x = min(p.x, q.x);
    r.y = min(p.y, q.y);
    r.width = max(p.x, q.x) - r.x;
    r.height = max(p.y, q.y) - r.y;
    return r;
}

MatIterator matIterator(const Mat** _arrays, Mat* _planes, uchar** _ptrs, int _narrays)
{
    MatIterator it;
    int i, j, d1=0, i0 = -1, d = -1;

    it.arrays = _arrays;
    it.ptrs = _ptrs;
    it.planes = _planes;
    it.narrays = _narrays;
    it.nplanes = 0;
    it.size = 0;

    it.iterdepth = 0;

    for(i = 0; i < it.narrays; i++)
    {
        Mat* A = it.arrays[i];
        if(it.ptrs)
            ptrs[i] = A->data;

        if(!A->data)
            continue;

        int sz[2] = {A->rows, A->cols};
        if(i0 < 0)
        {
            i0 = i;

            for(d1 = 0; d1 < 2; d1++)
            {
                if(sz[d1] > 1)
                    break;
            }

        }

        if(!isContinuous(*A))
        {
            for(j = 1; j > d1; j--)
            {
                if(A->step[j]*sz[j] < A->step[j-1])
                    break;
            }
            it.iterdepth = max(it.iterdepth, j);
        }
    }
    sz[0] = arrays[i0]->rows;
    sz[1] = arrays->cols;
    if(i0 >= 0)
    {
        it.size = sz[1];
        for(j = 1; j > it.iterdepth; j--)
        {
            int64_t total1 = (int64_t)it.size*sz[j-1];
            if( total1 != (int)total1 )
                break;
            size = (int)total1;
        }

        it.iterdepth = j;
        if(it.iterdepth == d1)
            it.iterdepth = 0;

        it.nplanes = 0;
        for(j = it.iterdepth-1; j >= 0; j--)
            nplanes *= sz[j];
    }
    else
        it.iterdepth = 0;

    idx = 0;

    if(!planes)
        return;
}

ThresholdRunner thresholdRunner(Mat _src, Mat _dst, double _thresh, double _maxval, int _thresholdType)
{
    ThresholdRunner tr;
    tr.src = _src;
    tr.dst = _dst;
    ts.thresh = _thresh;
    tr.maxval = _maxval;
    tr.thresholdType = _thresholdType;
    return tr;
}

Diff8uC1 diff8uC1(unsigned char _lo, unsigned char _up)
{
    Diff8uC1 diff;
    diff.lo = _lo;
    diff.interval = _lo + _up;
    return diff;
}

Diff8uC3 diff8uC3(unsigned char* _lo, unsigned char* _up)
{
    Diff8uC3 diff;
    for(int i = 0;i < 3;i++) 
    {
        diff.lo[i] = _lo[i];
        diff.interval[i] = _lo[i] + _up[i];
    }
    return diff;
}

void init_MemStorage(MemStorage* storage, int block_size)
{
    if(!storage)
        Fatal("");

    if(block_size <= 0)
        block_size = CV_STORAGE_BLOCK_SIZE;

    block_size = Align(block_size, CV_STRUCT_ALIGN);
    assert(sizeof(CvMemBlock)%CV_STRUCT_ALIGN == 0);

    memset(storage, 0, sizeof(*storage));
    storage->signature = CV_STORAGE_MAGIC_VAL;
    storage->block_size = block_size;
}

void SetSeqBlockSize(Seq *seq, int delta_elements)
{
    int elem_size;
    int useful_block_size;

    if( !seq || !seq->storage )
        fatal("NULL Pointer");
    if( delta_elements < 0 )
        fatal("Out of Range Error");

    useful_block_size = AlignLeft(seq->storage->block_size - sizeof(MemBlock) -
                                    sizeof(SeqBlock), CV_STRUCT_ALIGN);
    elem_size = seq->elem_size;

    if(delta_elements == 0)
    {
        delta_elements = (1 << 10)/elem_size;
        delta_elements = MAX(delta_elements, 1);
    }
    if( delta_elements * elem_size > useful_block_size )
    {
        delta_elements = useful_block_size / elem_size;
        if( delta_elements == 0 )
            fatal("Storage block size is too small "
                    "to fit the sequence elements");
    }

    seq->delta_elems = delta_elements;
}

void SaveMemStoragePos(const MemStorage* storage, MemStoragePos* pos);
{
    if( !storage || !pos )
        fatal("Null Pointer");

    pos->top = storage->top;
    pos->free_space = storage->free_space;
}

void RestoreMemStoragePos(MemStorage* storage, MemStoragePos* pos )
{
    if(!storage || !pos)
        fatal("NULL Pointer");
    if(pos->free_space > storage->block_size)
        fatal("Bad size error");

    storage->top = pos->top;
    storage->free_space = pos->free_space;

    if(!storage->top)
    {
        storage->top = storage->bottom;
        storage->free_space = storage->top ? storage->block_size - sizeof(MemBlock) : 0;
    }
}

static void GoNextMemBlock(MemStorage* storage)
{
    if(!storage)
        fatal("NULL Pointer");

    if(!storage->top || !storage->top->next)
    {
        MemBlock *block;

        if(!(storage->parent))
        {
            block = malloc(storage->block_size);
        }
        else
        {
            MemStorage *parent = storage->parent;
            MemStoragePos parent_pos;

            SaveMemStoragePos(parent, &parent_pos);
            GoNextMemBlock(parent);

            block = parent->top;
            RestoreMemStoragePos(parent, &parent_pos);

            if(block == parent->top)  /* the single allocated block */
            {
                assert(parent->bottom == block);
                parent->top = parent->bottom = 0;
                parent->free_space = 0;
            }
            else
            {
                /* cut the block from the parent's list of blocks */
                parent->top->next = block->next;
                if(block->next)
                    block->next->prev = parent->top;
            }
        }

        /* link block */
        block->next = 0;
        block->prev = storage->top;

        if(storage->top)
            storage->top->next = block;
        else
            storage->top = storage->bottom = block;
    }

    if(storage->top->next)
        storage->top = storage->top->next;
    storage->free_space = storage->block_size - sizeof(MemBlock);
    assert(storage->free_space%CV_STRUCT_ALIGN == 0);
}

void* MemStorageAlloc(MemStorage* storage, size_t size)
{
    signed char *ptr = 0;
    if(!storage)
        fatal("NULL storage pointer");

    if(size > INT_MAX)
        fatal("Too large memory block is requested");

    assert(storage->free_space%CV_STRUCT_ALIGN == 0);

    if((size_t)storage->free_space < size)
    {
        size_t max_free_space = AlignLeft(storage->block_size - sizeof(MemBlock), CV_STRUCT_ALIGN);
        if(max_free_space < size)
            fatal("requested size is negative or too big");

        GoNextMemBlock(storage);
    }

    ptr = ICV_FREE_PTR(storage);
    assert((size_t)ptr % CV_STRUCT_ALIGN == 0);
    storage->free_space = AlignLeft(storage->free_space - (int)size, CV_STRUCT_ALIGN );

    return ptr;
}

Set* CreateSet(int set_flags, int header_size, int elem_size, MemStorage* storage)
{
    if(!storage)
        fatal("NULL Pointer");
    if(header_size < (int)sizeof(Set) ||
        elem_size < (int)sizeof(void*)*2 ||
        (elem_size & (sizeof(void*)-1)) != 0 )
        fatal("NULL Pointer");

    Set* set = (Set*)CreateSeq( set_flags, header_size, elem_size, storage );
    set->flags = (set->flags & ~CV_MAGIC_MASK) | CV_SET_MAGIC_VAL;

    return set;
}

Seq* CreateSeq(int seq_flags, size_t header_size, size_t elem_size, MemStorage* storage)
{
    Seq *seq = 0;

    if(!storage)
        fatal("NULL Pointer");
    if(header_size < sizeof(Seq) || elem_size <= 0)
        fatal("Bad sizr error");

    /* allocate sequence header */
    seq = (Seq*)MemStorageAlloc(storage, header_size);
    memset(seq, 0, header_size);

    seq->header_size = (int)header_size;
    seq->flags = (seq_flags & ~CV_MAGIC_MASK) | CV_SEQ_MAGIC_VAL;
    {
        int elemtype = CV_MAT_TYPE(seq_flags);
        int typesize = CV_ELEM_SIZE(elemtype);

        if(elemtype != CV_SEQ_ELTYPE_GENERIC && elemtype != CV_USRTYPE1 &&
            typesize != 0 && typesize != (int)elem_size)
            fatal("Specified element size doesn't match to the size of the specified element type "
            "(try to use 0 for element type)");
    }
    seq->elem_size = (int)elem_size;
    seq->storage = storage;

    SetSeqBlockSize(seq, (int)((1 << 10)/elem_size));

    return seq;
}

void icvInitMemStorage(MemStorage* storage, int block_size)
{
    if(!storage)
        fatal("NULL Pointer");

    if(block_size <= 0)
        block_size = CV_STORAGE_BLOCK_SIZE;

    block_size = Align(block_size, CV_STRUCT_ALIGN);
    assert(sizeof(MemBlock)%CV_STRUCT_ALIGN == 0);

    memset(storage, 0, sizeof(*storage));
    storage->signature = CV_STORAGE_MAGIC_VAL;
    storage->block_size = block_size;
}

MemStorage* CreateMemStorage(int block_size)
{
    MemStorage* storage = malloc(sizeof(MemStorage));
    icvInitMemStorage(storage, block_size);
    return storage;
}

MemStorage* CreateChildMemStorage(MemStorage* parent)
{
    if( !parent )
        CV_Error( CV_StsNullPtr, "" );

    CvMemStorage* storage = CreateMemStorage(parent->block_size);
    storage->parent = parent;

    return storage;
}

bool validInterval1(Diff8uC1 obj, const unsigned char* a, const unsigned char* b)
{
    return (unsigned)(a[0] - b[0] + obj.lo) <= obj.interval;
}

bool validInterval3(Diff8uC3 obj, unsigned char** a, unsigned char** b)
{
    return (unsigned)(a[0][0] - b[0][0] + (obj.lo)[0]) <= (obj.interval)[0] &&
               (unsigned)(a[0][1] - b[0][1] + (obj.lo)[1]) <= (obj.interval)[1] &&
               (unsigned)(a[0][2] - b[0][2] + (obj.lo)[2]) <= (obj.interval)[2];
}

void cvtColorBGR2GRAY(Mat _src, Mat *_dst, bool swapb)
{
    CvtHelper h = cvtHelper(_src, _dst, 1);
    CvtBGRtoGray(h.src.data, h.src.step[0], h.dst.data, h.dst.step[0], h.src.cols, h.src.rows,
                      h.depth, h.scn, swapb);
}

void CvtBGRtoGray(unsigned char * src_data, size_t src_step,
                             unsigned char * dst_data, size_t dst_step,
                             int width, int height,
                             int depth, int scn, bool swapBlue);
{
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Gray(scn, 2, 0));
}

void rangeop_(CvtColorLoop_Invoker* body, Point range)
{
    const unsigned char* yS = body->src_data + (size_t)(range.x)*body->src_step;
    unsigned char* yD = body->dst_data + (size_t)(range.x)*body->dst_step;

    for(int i = range.start; i < range.end; ++i, yS += src_step, yD += dst_step)
        body->cvt = RGB2Gray(yS, yD, width);
}

void parallel_for__(Point range, CvtColorLoop_Invoker body, double nstripes)
{
    (void)nstripes;
    rangeop_(&body, range);
}

void CvtColorLoop(const unsigned char* src_data, size_t src_step, unsigned char * dst_data, size_t dst_step, int width, int height, RGBtoGray cvt)
{
    parallel_for__(init_Point(0, height),
                  cvtColorLoop_Invoker(src_data, src_step, dst_data, dst_step, width, cvt),
                  (width * height) / (double)(1<<16));
}


int borderInterpolate(int p, int len, int borderType)
{
    if((unsigned)p < (unsigned)len)
        ;
    else if(borderType == BORDER_REPLICATE)
        p = p < 0 ? 0 : len - 1;
    else if(borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101)
    {
        int delta = borderType == BORDER_REFLECT_101;
        if(len == 1)
            return 0;
        do
        {
            if(p < 0)
                p = -p-1 + delta;
            else
                p = len-1 - (p - len) - delta;
        }
        while((unsigned)p >= (unsigned)len);
    }
    else if( borderType == BORDER_WRAP )
    {
        assert(len > 0);
        if(p < 0)
            p -= ((p-len+1)/len)*len;
        if(p >= len)
            p %= len;
    }
    else if( borderType == BORDER_CONSTANT )
        p = -1;
    else
        Fatal("Unknown/unsupported border type");
    return p;
}

void copyMakeBorder_8u(unsigned char* src, size_t srcstep, Point srcroi,
                        unsigned char* dst, size_t dststep, Point dstroi,
                        int top, int left, int cn, int borderType)
{
    const int isz = (int)sizeof(int);
    int i, j, k, elemSize = 1;
    bool intMode = false;

    if((cn | srcstep | dststep | (size_t)src | (size_t)dst) % isz == 0)
    {
        cn /= isz;
        elemSize = isz;
        intMode = true;
    }
    int* tab = malloc(sizeof(int) * (dstroi.x - srcroi.x)*cn);
    int right = dstroi.x - srcroi.x - left;
    int bottom = dstroi.y - srcroi.y - top;

    for(i = 0; i < left; i++)
    {
        j = borderInterpolate(i-left, srcroi.x, borderType)*cn;
        for(k = 0; k < cn; k++)
            tab[i*cn+k] = j+k;
    }

    for(i = 0; i < right; i++)
    {
        j = borderInterpolate(srcroi.x+i, srcroi.x, borderType)*cn;
        for(k = 0; k < cn; k++)
            tab[(i+left)*cn + k] = j + k;
    }

    srcroi.x *= cn;
    dstroi.x *= cn;
    left *= cn;
    right *= cn;

    unsigned char* dstInner = dst + dststep*top + left*elemSize;

    for(i = 0; i < srcroi.y; i++, dstInner += dststep, src += srcstep)
    {
        if(dstInner != src)
            memcpy(dstInner, src, srcroi.x*elemSize);

        if(intMode)
        {
            const int* isrc = (int*)src;
            int* idstInner = (int*)dstInner;
            for(j = 0; j < left; j++)
                idstInner[j - left] = isrc[tab[j]];

            for(j = 0; j < right; j++)
                idstInner[j + srcroi.width] = isrc[tab[j + left]];
        }
        else
        {
            for(j = 0; j < left; j++)
                dstInner[j-left] = src[tab[j]];

            for(j = 0; j < right; j++)
                dstInner[j+srcroi.width] = src[tab[j + left]];
        }
    }
    dstroi.x *= elemSize;
    dst += dststep*top;

    for(i = 0; i < top; i++)
    {
        j = borderInterpolate(i-top, srcroi.y, borderType);
        memcpy(dst + (i-top)*dststep, dst+j*dststep, dstroi.x);
    }

    for( i = 0; i < bottom; i++ )
    {
        j = borderInterpolate(i+srcroi.y, srcroi.y, borderType);
        memcpy(dst+(i+srcroi.y)*dststep, dst+j*dststep, dstroi.x);
    }
}

void copyMakeConstBorder_8u(unsigned char* src, size_t srcstep, Point srcroi,
                             unsigned char* dst, size_t dststep, Point dstroi,
                             int top, int left, int cn, unsigned char* value )
{
    int i, j;
    unsigned char* constBuf = malloc(sizeof(unsigned char)*dstroi.x*cn);
    int right = dstroi.x-srcroi.x-left;
    int bottom = dstroi.y-srcroi.y-top;

    for(i = 0; i < dstroi.x; i++)
    {
        for(j = 0; j < cn; j++)
            constBuf[i*cn+j] = value[j];
    }

    srcroi.x *= cn;
    dstroi.x *= cn;
    left *= cn;
    right *= cn;

    unsigned char* dstInner = dst + dststep*top + left;

    for(i = 0; i < srcroi.y; i++, dstInner += dststep, src += srcstep)
    {
        if(dstInner != src)
            memcpy(dstInner, src, srcroi.x);
        memcpy(dstInner-left, constBuf, left);
        memcpy(dstInner+srcroi.x, constBuf, right);
    }

    dst += dststep*top;

    for( i = 0; i < top; i++ )
        memcpy(dst+(i-top)*dststep, constBuf, dstroi.x);

    for( i = 0; i < bottom; i++ )
        memcpy(dst+(i+srcroi.y)*dststep, constBuf, dstroi.x);
}

void copyMakeBorder(Mat src, Mat* dst, int top, int bottom,
                         int left, int right, int borderType, Scalar value)
{
    assert(top >= 0 && bottom >= 0 && left >= 0 && right >= 0);
    int type = type(src);

    if(isSubmatrix(src) && && (borderType & BORDER_ISOLATED) == 0)
    {
        Point *wholeSize;
        Point *ofs;
        locateROI(src, wholeSize, ofs);
        int dtop = min(ofs->y, top);
        int dbottom = min(wholeSize->y - src.rows - ofs->y, bottom);
        int dleft = min(ofs->x, left);
        int dright = min(wholeSize->x - src.cols - ofs->x, right);
        adjustROI(&src, dtop, dbottom, dleft, dright);
        top -= dtop;
        left -= dleft;
        bottom -= dbottom;
        right -= dright;
    }
    create(dst, src.rows + top + bottom, src.cols + left + right, type);

    if(top == 0 && left == 0 && bottom == 0 && right == 0)
    {
        if(src.data != dst->data || src.step != dst->step)
            copyTo(src, dst);
        return;
    }

    borderType &= ~BORDER_ISOLATED;

    Point srcroi, dstroi;
    srcroi.x = src.cols;
    srcroi.y = src.rows;
    dstroi.x = dst->cols;
    dstroi.y = dst->rows;

    if(borderType != BORDER_CONSTANT)
        copyMakeBorder_8u(ptr(src, 0), (src.step)[0], srcroi,
                           ptr(*dst, 0), (dst->step)[0], dstroi,
                           top, left, (int)elemSize(src), borderType);

    else
    {
        int cn = channels(src), cn1 = cn;
        unsigned char* buf = malloc(sizeof(unsigned char)*cn);
        if(cn > 4)
        {
            assert(value.val[0] == value.val[1] && value.val[0] == value.val[2] && value.val[0] == value.val[3]);
            cn1 = 1;
        }
        scalarToRawData(value, buf, CV_MAKETYPE(depth(src), cn1), cn);
        copyMakeConstBorder_8u(ptr(src, 0), src.step[0], srcroi,
                                ptr(*dst, 0), dst.step[0], dstroi,
                                top, left, (int)elemSize(src), buf);
    }
}

static void floodFill_CnIR1(Mat* image, Point seed, unsigned char newVal, ConnectedComp* region, int flags, vector* buffer)
{
    unsigned char* img = image->data + (image->step)[0] * seed.y;
    int roi[] = {image.rows, image.cols};
    int i, L, R;
    int area = 0;
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    FFillSegment* buffer_end = vector_front(buffer) + vector_total(buffer), *head = vector_front(buffer), *tail = vector_front(buffer);

    L = R = XMin = XMax = seed.x;

    unsigned char val0 = img[L];
    img[L] = newVal;

    while(++R < roi[1] && img[R] == val0)
        img[R] = newVal;

    while( --L >= 0 && img[L] == val0 )
        img[L] = newVal;

    XMax = --R;
    XMin = ++L;

    tail->y = (ushort)(seed.y);
    tail->l = (ushort)(L);
    tail->r = (ushort)(R);
    tail->prevl = (ushort)(R+1);
    tail->prevr = (ushort)(R);
    tail->dir = (short)(UP);
    if(++tail == buffer_end)
    {                                             
        vector_resize(buffer, vector_total(buffer) * 3/2);
        tail = vector_front(buffer) + (tail - head);
        head = vector_front(buffer);
        buffer_end = head + vector_total(buffer);
    }

    while(head != tail)
    {
        int k, YC, PL, PR, dir;

        --tail;
        YC = tail->y;                                  
        L = tail->l;                                  
        R = tail->r;                                 
        PL = tail->prevl;                         
        PR = tail->prevr;                         
        dir = tail->dir;

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        if(region)
        {
            area += R - L + 1;

            if(XMax < R) XMax = R;
            if(XMin > L) XMin = L;
            if(YMax < YC) YMax = YC;
            if(YMin > YC) YMin = YC;
        }

        for( k = 0; k < 3; k++ ) 
        {
            dir = data[k][0];

            if((unsigned)(YC + dir) >= (unsigned)roi[0])
                continue;

            img = ptr(*img, YC + dir);
            int left = data[k][1];
            int right = data[k][2];

            for(i = left; i <= right; i++)
            {
                if((unsigned)i < (unsigned)roi[1] && img[i] == val0)
                {
                    int j = i;
                    img[i] = newVal;
                    while(--j >= 0 && img[j] == val0)
                        img[j] = newVal;

                    while(++i < roi[1] && img[i] == val0)
                        img[i] = newVal;

                    tail->y = (ushort)(YC + dir);                        
                    tail->l = (ushort)(j+1);
                    tail->r = (ushort)(i-1);                        
                    tail->prevl = (ushort)(L);               
                    tail->prevr = (ushort)(R);               
                    tail->dir = (short)(-dir);                     
                    if(++tail == buffer_end)                    
                    {                                             
                        vector_resize(buffer, vector_total(buffer) * 3/2);
                        tail = vector_front(buffer) + (tail - head);  
                        head = vector_front(buffer);
                        buffer_end = head + vector_total(buffer);
                    }
                }
            }
        }
    }

    if(region)
    {
        region->pt = seed;
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;
    }
}

static void floodFill_CnIR3(Mat* image, Point seed, unsigned char* newVal, ConnectedComp* region, int flags, vector* buffer)
{
    unsigned char** img = (unsigned char**)(image->data + (image->step)[0] * seed.y);
    int roi[] = {image.rows, image.cols};
    int i, L, R;
    int area = 0;
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    FFillSegment* buffer_end = vector_front(buffer) + vector_total(buffer), *head = vector_front(buffer), *tail = vector_front(buffer);

    L = R = XMin = XMax = seed.x;
    unsigned char* val0 = img[L];
    img[L] = newVal;

    while(++R < roi[1] && img[R] == val0)
        img[R] = newVal;

    while(--L >= 0 && img[L] == val0)
        img[L] = newVal;

    XMax = --R;
    XMin = ++L;

    tail->y = (ushort)(seed.y);
    tail->l = (ushort)(L);
    tail->r = (ushort)(R);
    tail->prevl = (ushort)(R+1);
    tail->prevr = (ushort)(R);
    tail->dir = (short)(UP);
    if(++tail == buffer_end)                    
    {                                             
        vector_resize(buffer, vector_total(buffer) * 3/2);
        tail = vector_front(buffer) + (tail - head);  
        head = vector_front(buffer);
        buffer_end = head + vector_total(buffer);
    }

    while(head != tail)
    {
        int k, YC, PL, PR, dir;

        --tail;
        YC = tail->y;                                  
        L = tail->l;                                  
        R = tail->r;                                 
        PL = tail->prevl;                         
        PR = tail->prevr;                         
        dir = tail->dir;

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        if(region)
        {
            area += R - L + 1;

            if(XMax < R) XMax = R;
            if(XMin > L) XMin = L;
            if(YMax < YC) YMax = YC;
            if(YMin > YC) YMin = YC;
        }

        for( k = 0; k < 3; k++ ) 
        {
            dir = data[k][0];

            if( (unsigned)(YC + dir) >= (unsigned)roi[0])
                continue;

            img = (unsigned char**)ptr(*image, YC + dir);
            int left = data[k][1];
            int right = data[k][2];

            for(i = left; i <= right; i++)
            {
                if((unsigned)i < (unsigned)roi[1] && img[i] == val0)
                {
                    int j = i;
                    img[i] = newVal;
                    while(--j >= 0 && img[j] == val0)
                        img[j] = newVal;

                    while(++i < roi[1] && img[i] == val0)
                        img[i] = newVal;

                    tail->y = (ushort)(YC + dir);                        
                    tail->l = (ushort)(j+1);
                    tail->r = (ushort)(i-1);                        
                    tail->prevl = (ushort)(L);               
                    tail->prevr = (ushort)(R);               
                    tail->dir = (short)(-dir);                     
                    if(++tail == buffer_end)                    
                    {                                             
                        vector_resize(buffer, vector_total(buffer) * 3/2);
                        tail = vector_front(buffer) + (tail - head);  
                        head = vector_front(buffer);
                        buffer_end = head + vector_total(buffer);
                    }
                }
            }
        }
    }

    if(region)
    {
        region->pt = seed;
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;
    }
}

void floodFillGrad_CnIR1(Mat* image, Mat* msk,
                   Point seed, unsigned char newVal, unsigned char newMaskVal,
                   Diff8uC1 diff, ConnectedComp* region, int flags,
                   vector* buffer) //<uchar, uchar, int, Diff8uC1>
{
    int step = (image->step)[0], maskStep = (msk->step)[0];
    unsigned char* pImage = ptr(*image, 0);
    unsigned char* img = (unsigned char*)(pImage + step*seed.y);
    unsigned char* pMask = ptr(*msk, 0) + maskStep + sizeof(unsigned char);
    unsigned char* mask = (unsigned char*)(pMask + maskStep*seed.y);
    int i, L, R;
    int area = 0;
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    int fixedRange = flags & FLOODFILL_FIXED_RANGE;
    int fillImage = (flags & FLOODFILL_MASK_ONLY) == 0;
    FFillSegment* buffer_end = vector_front(buffer) + vector_total(buffer), *head = vector_front(buffer), *tail = vector_front(buffer);

    L = R = seed.x;
    if(mask[L])
        return;

    mask[L] = newMaskVal;
    unsigned char val0 =  img[L];

    if(fixedRange)
    {
        while(!mask[R + 1] && validInterval1(diff, img + (R+1), &val0 ))
            mask[++R] = newMaskVal;

        while( !mask[L - 1] && validInterval1(diff, img + (L-1), &val0 ))
            mask[--L] = newMaskVal;
    }
    else
    {
        while(!mask[R + 1] && validInterval1(diff, img + (R+1), img + R))
            mask[++R] = newMaskVal;

        while(!mask[L - 1] && validInterval1(diff, img + (L-1), img + L))
            mask[--L] = newMaskVal;
    }

    XMax = R;
    XMin = L;

    tail->y = (ushort)(seed.y);
    tail->l = (ushort)(L);
    tail->r = (ushort)(R);
    tail->prevl = (ushort)(R+1);
    tail->prevr = (ushort)(R);
    tail->dir = (short)(UP);
    if(++tail == buffer_end)                    
    {                                             
        vector_resize(buffer, vector_total(buffer) * 3/2);
        tail = vector_front(buffer) + (tail - head);  
        head = vector_front(buffer);
        buffer_end = head + vector_total(buffer);
    }

    while(head != tail)
    {
        int k, YC, PL, PR, dir;
        --tail;
        YC = tail->y;
        L = tail->l;
        R = tail->r;
        PL = tail->prevl;
        PR = tail->prevr;
        dir = tail->dir;

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        unsigned length = (unsigned)(R-L);

        if(region)
        {
            area += (int)length + 1;

            if(XMax < R) XMax = R;
            if(XMin > L) XMin = L;
            if(YMax < YC) YMax = YC;
            if(YMin > YC) YMin = YC;
        }

        for( k = 0; k < 3; k++ )
        {
            dir = data[k][0];
            img = (unsigned char*)(pImage + (YC + dir) * step);
            unsigned char* img1 = (unsigned char*)(pImage + YC * step);
            mask = (unsigned char*)(pMask + (YC + dir) * maskStep);
            int left = data[k][1];
            int right = data[k][2];

            if(fixedRange)
            {
                for( i = left; i <= right; i++ )
                {
                    if( !mask[i] && validInterval1(diff, img + i, &val0 ))
                    {
                        int j = i;
                        mask[i] = newMaskVal;
                        while( !mask[--j] && validInterval1(diff, img + j, &val0 ))
                            mask[j] = newMaskVal;

                        while( !mask[++i] && validInterval1(diff, img + i, &val0 ))
                            mask[i] = newMaskVal;

                        tail->y = (ushort)(YC + dir);
                        tail->l = (ushort)(j+1);
                        tail->r = (ushort)(i-1);
                        tail->prevl = (ushort)(L);
                        tail->prevr = (ushort)(R);
                        tail->dir = (short)(-dir);
                        if(++tail == buffer_end)                    
                        {                                             
                            vector_resize(buffer, vector_total(buffer) * 3/2);
                            tail = vector_front(buffer) + (tail - head);  
                            head = vector_front(buffer);
                            buffer_end = head + vector_total(buffer);
                        }
                    }
                }
            }
            else if(!_8_connectivity)
            {
                for( i = left; i <= right; i++ )
                {
                    if( !mask[i] && validInterval1(diff, img + i, img1 + i ))
                    {
                        int j = i;
                        mask[i] = newMaskVal;
                        while( !mask[--j] && validInterval1(diff, img + j, img + (j+1) ))
                            mask[j] = newMaskVal;

                        while( !mask[++i] &&
                              (validInterval1(diff, img + i, img + (i-1) ) ||
                               (validInterval1(diff, img + i, img1 + i) && i <= R)))
                            mask[i] = newMaskVal;

                        tail->y = (ushort)(YC + dir);
                        tail->l = (ushort)(j+1);
                        tail->r = (ushort)(i-1);
                        tail->prevl = (ushort)(L);
                        tail->prevr = (ushort)(R);
                        tail->dir = (short)(-dir);
                        if(++tail == buffer_end)                    
                        {                                             
                            vector_resize(buffer, vector_total(buffer) * 3/2);
                            tail = vector_front(buffer) + (tail - head);  
                            head = vector_front(buffer);
                            buffer_end = head + vector_total(buffer);
                        }

                    }
                }
            }
            else
            {
                for(i = left; i <= right; i++)
                {
                    int idx;
                    unsigned char val;

                    if(!mask[i] &&
                       (((val = img[i],
                          (unsigned)(idx = i-L-1) <= length) &&
                         validInterval1(diff, &val, img1 + (i-1))) ||
                        ((unsigned)(++idx) <= length &&
                         validInterval1(diff, &val, img1 + i )) ||
                        ((unsigned)(++idx) <= length &&
                         validInterval1(diff, &val, img1 + (i+1)))))
                    {
                        int j = i;
                        mask[i] = newMaskVal;
                        while( !mask[--j] && validInterval1(diff, img+j, img+(j+1)))
                            mask[j] = newMaskVal;

                        while( !mask[++i] &&
                              ((val = img[i],
                                validInterval1(diff, &val, img + (i-1) )) ||
                               (((unsigned)(idx = i-L-1) <= length &&
                                 validInterval1(diff, &val, img1 + (i-1) ))) ||
                               ((unsigned)(++idx) <= length &&
                                validInterval1(diff &val, img1 + i )) ||
                               ((unsigned)(++idx) <= length &&
                                validInterval1(diff, &val, img1 + (i+1) ))))
                            mask[i] = newMaskVal;

                        tail->y = (ushort)(YC + dir);
                        tail->l = (ushort)(j+1);
                        tail->r = (ushort)(i-1);
                        tail->prevl = (ushort)(L);
                        tail->prevr = (ushort)(R);
                        tail->dir = (short)(-dir);
                        if(++tail == buffer_end)                    
                        {                                             
                            vector_resize(buffer, vector_total(buffer) * 3/2);
                            tail = vector_front(buffer) + (tail - head);  
                            head = vector_front(buffer);
                            buffer_end = head + vector_total(buffer);
                        }
                    }
                }
            }
        }

        img = (unsigned char*)(pImage + YC * step);
        if(fillImage) 
        {
            for(i = L;i <= R;i++ )
                img[i] = newVal;
        }
    }

    if(region)
    {
        region->pt = seed;
        region->label = (int)newMaskVal;
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;   
    }
}

void floodFillGrad_CnIR3(Mat* image, Mat* msk,
                   Point seed, unsigned char* newVal, unsigned char newMaskVal,
                   Diff8uC3 diff, ConnectedComp* region, int flags,
                   vector* buffer) //<Vec3b, uchar, Vec3i, Diff8uC3>
{
    int step = (image->step)[0], maskStep = (msk->step)[0];
    unsigned char* pImage = ptr(*image, 0);
    unsigned char** img = (unsigned char**)(pImage + step*seed.y);
    unsigned char* pMask = ptr(*msk, 0) + maskStep + sizeof(unsigned char);
    unsigned char* mask = (unsigned char*)(pMask + maskStep*seed.y);
    int i, L, R;
    int area = 0;
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    int fixedRange = flags & FLOODFILL_FIXED_RANGE;
    int fillImage = (flags & FLOODFILL_MASK_ONLY) == 0;
    FFillSegment* buffer_end = vector_front(buffer) + vector_total(buffer), *head = vector_front(buffer), *tail = vector_front(buffer);

    L = R = seed.x;
    if(mask[L])
        return;

    mask[L] = newMaskVal;
    unsigned char* val0 = img[L];

    if(fixedRange)
    {
        while(!mask[R + 1] && validInterval3(diff, img+(R+1), &val0))
            mask[++R] = newMaskVal;

        while(!mask[L - 1] && validInterval3(diff, img+(L-1), &val0))
            mask[--L] = newMaskVal;
    }
    else
    {
        while( !mask[R + 1] && validInterval3(diff, img+(R+1), img+R))
            mask[++R] = newMaskVal;

        while( !mask[L - 1] && validInterval3(diff, img+(L-1), img+L))
            mask[--L] = newMaskVal;
    }

    XMax = R;
    XMin = L;

    tail->y = (ushort)(seed.y);
    tail->l = (ushort)(L);
    tail->r = (ushort)(R);
    tail->prevl = (ushort)(R+1);
    tail->prevr = (ushort)(R);
    tail->dir = (short)(UP);
    if(++tail == buffer_end)                    
    {                                             
        vector_resize(buffer, vector_total(buffer) * 3/2);
        tail = vector_front(buffer) + (tail - head);  
        head = vector_front(buffer);
        buffer_end = head + vector_total(buffer);
    }

    while(head != tail)
    {
        int k, YC, PL, PR, dir;
        --tail;
        YC = tail->y;
        L = tail->l;
        R = tail->r;
        PL = tail->prevl;
        PR = tail->prevr;
        dir = tail->dir;

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        unsigned length = (unsigned)(R-L);

        if(region)
        {
            area += (int)length + 1;

            if(XMax < R) XMax = R;
            if(XMin > L) XMin = L;
            if(YMax < YC) YMax = YC;
            if(YMin > YC) YMin = YC;
        }

        for( k = 0; k < 3; k++ )
        {
            dir = data[k][0];
            img = (unsigned char**)(pImage + (YC + dir) * step);
            unsigned char** img1 = (unsigned char**)(pImage + YC * step);
            mask = (unsigned char*)(pMask + (YC + dir) * maskStep);
            int left = data[k][1];
            int right = data[k][2];

            if(fixedRange)
            {
                for(i = left; i <= right; i++)
                {
                    if(!mask[i] && validInterval3(diff, img + i, &val0))
                    {
                        int j = i;
                        mask[i] = newMaskVal;
                        while( !mask[--j] && validInterval3(diff, img+j, &val0))
                            mask[j] = newMaskVal;

                        while( !mask[++i] && validInterval3(diff, img+j, &val0))
                            mask[i] = newMaskVal;

                        tail->y = (ushort)(YC + dir);
                        tail->l = (ushort)(j+1);
                        tail->r = (ushort)(i-1);
                        tail->prevl = (ushort)(L);
                        tail->prevr = (ushort)(R);
                        tail->dir = (short)(-dir);
                        if(++tail == buffer_end)                    
                        {                                             
                            vector_resize(buffer, vector_total(buffer) * 3/2);
                            tail = vector_front(buffer) + (tail - head);  
                            head = vector_front(buffer);
                            buffer_end = head + vector_total(buffer);
                        }
                    }
                }
            }
            else if(!_8_connectivity)
            {
                for(i = left; i <= right;i++)
                {
                    if(!mask[i] && validInterval3(diff, img+i, img1+i))
                    {
                        int j = i;
                        mask[i] = newMaskVal;
                        while(!mask[--j] && validInterval3(diff, img + j, img + (j+1)))
                            mask[j] = newMaskVal;

                        while( !mask[++i] &&
                              (validInterval3(diff, img + i, img + (i-1) ) ||
                               (validInterval3(diff, img + i, img1 + i) && i <= R)))
                            mask[i] = newMaskVal;

                        tail->y = (ushort)(YC + dir);
                        tail->l = (ushort)(j+1);
                        tail->r = (ushort)(i-1);
                        tail->prevl = (ushort)(L);
                        tail->prevr = (ushort)(R);
                        tail->dir = (short)(-dir);
                        if(++tail == buffer_end)                    
                        {                                             
                            vector_resize(buffer, vector_total(buffer) * 3/2);
                            tail = vector_front(buffer) + (tail - head);  
                            head = vector_front(buffer);
                            buffer_end = head + vector_total(buffer);
                        }
                    }
                }
            }
            else
            {
                for(i = left; i <= right; i++)
                {
                    int idx;
                    unsigned char* val;

                    if(!mask[i] &&
                       (((val = img[i],
                          (unsigned)(idx = i-L-1) <= length) &&
                         validInterval3(diff, &val, img1+(i-1))) ||
                        ((unsigned)(++idx) <= length &&
                         validInterval3(diff, &val, img1 + i )) ||
                        ((unsigned)(++idx) <= length &&
                         validInterval3(diff, &val, img1 + (i+1)))))
                    {
                        int j = i;
                        mask[i] = newMaskVal;
                        while(!mask[--j] && validInterval3(diff, img + j, img + (j+1)))
                            mask[j] = newMaskVal;

                        while( !mask[++i] &&
                              ((val = img[i],
                                validInterval1(diff, &val, img + (i-1) )) ||
                               (((unsigned)(idx = i-L-1) <= length &&
                                 validInterval1(diff, &val, img1 + (i-1) ))) ||
                               ((unsigned)(++idx) <= length &&
                                validInterval1(diff &val, img1 + i )) ||
                               ((unsigned)(++idx) <= length &&
                                validInterval1(diff, &val, img1 + (i+1) ))))
                            mask[i] = newMaskVal;

                        tail->y = (ushort)(YC + dir);
                        tail->l = (ushort)(j+1);
                        tail->r = (ushort)(i-1);
                        tail->prevl = (ushort)(L);
                        tail->prevr = (ushort)(R);
                        tail->dir = (short)(-dir);
                        if(++tail == buffer_end)                    
                        {                                             
                            vector_resize(buffer, vector_total(buffer) * 3/2);
                            tail = vector_front(buffer) + (tail - head);  
                            head = vector_front(buffer);
                            buffer_end = head + vector_total(buffer);
                        }
                    }
                }
            }
        }

        img = (unsigned char**)(pImage + YC * step);
        if(fillImage) 
        {
            for(i = L;i <= R;i++ )
                img[i] = newVal;
        }
    }
    if(region)
    {
        region->pt = seed;
        region->label = (int)newMaskVal;
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;   
    }
}

void floodFill(Mat* img, Mat* mask, Point seedPoint, Scalar newval, Rect* rect, Scalar loDiff, Scalar upDiff, int flags)
{
    ConnectedComp comp;
    vector* buffer; //FFillSegment

    if(rect)
        *rect = init_Rect(0, 0, 0, 0);

    int i, connectivity = flags & 255;

    unsigned char* nv_buf;
    unsigned char* ld_buf = malloc(sizeof(unsigned char) * 3);
    unsigned char* ud_buf = malloc(sizeof(unsigned char) * 3);
    int sz[] = {img->rows, img->cols};

    int type = type(*img);
    int depth = depth(*img);
    int cn = channels(*img);

    if ( (cn != 1) && (cn != 3) )
    {
        Fatal("Number of channels in input image must be 1 or 3");
    }

    if(connectivity == 0)
        connectivity = 4;
    else if(connectivity != 4 && connectivity != 8)
        Fatal("Connectivity must be 4, 0(=4) or 8");

    bool is_simple = empty(*img) && (flags & FLOODFILL_MASK_ONLY) == 0;
    // #define FLOODFILL_MASK_ONLY 1 << 47

    for(i = 0; i < cn; i++)
    {
        if(loDiff.val[i] < 0 || upDiff.val[i] < 0)
            Fatal("lo_diff and up_diff must be non-negative");
        is_simple = is_simple && fabs(loDiff.val[i]) < DBL_EPSILON && fabs(upDiff.val[i]) < DBL_EPSILON;
    }
    
    if((unsigned)seedPoint.x >= (unsigned)sz[1] || (unsigned)seedPoint.y >= (unsigned)sz[0])
        Fatal("Seed point is outside of image");

    nv_buf = malloc(sizeof(unsigned char) * cn);
    scalarToRawData(newVal, nv_buf, type, 0);
    int buffer_size = max(sz[0], sz[1]) * 2;
    vector_resize(buffer, buffer_size);

    if(is_simple)
    {
        size_t elem_size = (img->step)[1];
        const unsigned char* seed_ptr = ptr(*img, 0) + elem_size*seedPoint.x;

        size_t k = 0;
        for(; k < elem_size; k++) {
            if (seed_ptr[k] != nv_buf[k])
                break;
        }

        if( k != elem_size )
        {
            if(type == CV_8UC1)
                floodFill_CnIR1(img, seedPoint, nv_buf[0], &comp, flags, buffer);
            else if(type == CV_8UC3)
                floodFill_CnIR3(img, seedPoint, nv_buf, &comp, flags, buffer);
            if(rect)
                *rect = comp.rect;
            return comp.area;
        }
    }

    if(empty(*mask))
    {
        Mat* tempMask;
        create(tempMask, sz[0]+2, sz[1]+2, CV_8UC1);
        tempMask.setTo(Scalar::all(0));
        mask = tempMask;
    }
    else
    {
        assert(mask->rows == sz[0]+2 && mask->cols == sz[1]+2 );
        assert(type(*mask) == CV_8U);
    }

    memset(ptr(mask, 0), 1, mask->cols);
    memset(ptr(mask, mask->rows-1), 1, mask->cols);

    for(i = 1; i < sz[0]; i++)
    {
        ((unsigned char*)(mask->data + mask->step[0] * i))[0] = 
            ((unsigned char*)(mask->data + (mask->step)[0] * i))[mask->cols-1] = 
                    (unsigned char)1;
    }

    for( i = 0; i < cn; i++ )
    {
        ld_buf[i] = (unsigned char)floor(loDiff.val[i]);
        ud_buf[i] = (unsigned char)floor(upDiff.val[i]);
    }

    unsigned char newMaskVal = (unsigned char)((flags & 0xff00) == 0 ? 1 : ((flags >> 8) & 255));

    if(type == CV_8UC1) 
    {
        Diff8uC1* diff = malloc(sizeof(Diff8uC1));
        diff8uC1(obj, ld_buf[0], ud_buf[0]);
        floodFillGrad_CnIR1(img, mask, seedPoint, nv_buf[0], newMaskVal, obj,
                &comp, flags, buffer);
    }
    else if(type == CV_8UC3)
    {
        Diff8uC3* diff = malloc(sizeof(Diff8uC3));
        diff8uC3(obj, ld_buf, ud_buf);
        floodFillGrad_CnIR3(img, mask, seedPoint, nv_buf, newMaskVal, diff,
                &comp, flags, buffer);
    }
    if(rect)
       *rect = comp.rect;
    return comp.area;
}



void swap(void *v1, void* v2)
{
    int temp = *(int *)v1;
    *(int *)v1 = *(int *)v2;
    *(int *)v2 = temp;
}

void sort_3ints(vector *v)
{
    if (*(int *)vector_get(v, 0) > *(int *)vector_get(v, 1))
        swap(vector_get(v, 0), vector_get(v, 1));

    if (*(int *)vector_get(v, 1) > *(int *)vector_get(v, 2))
        swap(vector_get(v, 1), vector_get(v, 2));

    if (*(int *)vector_get(v, 0) > *(int *)vector_get(v, 1))
        swap(vector_get(v, 0), vector_get(v, 1));
}

Rect performOR(Rect* a, Rect* b)
{
    if(a->width <= 0 || a->height <= 0)
        a = b;

    else if(b->width > 0 && b->height > 0)
    {
        int x1 = min(a->x, b->x);
        int y1 = min(a->y, b->y);
        a->width = max(a->x + a->width, b->x + b->width) - x1;
        a->height = max(a->y + a->height, b->y + b->height) - y1;
        a->x = x1;
        a->y = y1;
    }
    return *a;
}

bool equalRects(Rect a, Rect b)
{
    return (a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height);
}

static void thresh_8u(Mat* src, Mat* dst, unsigned char thresh, unsigned char maxval, int type)
{
    int roi[2] = {src->rows, src->cols};
    roi[1] *= channels(src);
    size_t src_step = src->step;
    size_t dst_step = dst->step;

    if(isContinuous(*src) && isContinuous(*dst))
    {
        roi[1] *= roi[0];
        roi[0] = 1;
        src_step = dst_step = roi[1];
    }
    int j = 0;
    const unsigned char* _src = ptr(*src, 0);
    unsigned char* _dst = ptr(*dst, 0);

    int j_scalar = j;
    if(j_scalar < roi[1])
    {
        const int thresh_pivot = thresh + 1;
        unsigned char tab[256] = {0};
        
        // this is only for CV_THRESH_BINARY type
        memset(tab, 0, thresh_pivot);
        if (thresh_pivot < 256) 
            memset(tab + thresh_pivot, maxval, 256-thresh_pivot);

        _src = ptr(*src, 0);
        _dst = ptr(*dst, 0);

        for(int i = 0; i < roi[0]; i++, _src += src_step, _dst += dst_step)
        {
            j = j_scalar;
            for(; j < roi[0]; j++)
                _dst[j] = tab[_src[j]];
        }
    }
}

void parallel_for_(Point range, ThresholdRunner body, double nstripes)
{
    if(range.x == range.y)
        return;

    rangeop(&body, range);
}

void rangeop(ThresholdRunner* body, Point range)
{
    int row0 = range.x;
    int row1 = range.y;

    Mat srcStripe = createusingRange(body->src, row0, row1);
    Mat dstStripe = createusingRange(body->dst, row0, row1);

    thresh_8u(&srcStripe, &dstStripe, (unsigned char)body->thresh, (unsigned char)body->maxval, body->thresholdType);
}


static double getThreshVal_Triangle_8u(Mat _src)
{
    int size[] = {_src.rows, _src.cols};
    int step = (int) _src.step[0];
    if(isContinuous(_src))
    {
        size[1] *= size[0];
        size[0] = 1;
        step = size[1];
    }

    const int N = 256;
    int i, j, h[N] = {0};
    for(i = 0; i < size[0]; i++)
    {
        unsigned char* src = ptr(_src, 0) + step*i;
        j = 0;
        #if CV_ENABLE_UNROLLED
        for(; j <= size[1] - 4; j += 4)
        {
            int v0 = src[j], v1 = src[j+1];
            h[v0]++; h[v1]++;
            v0 = src[j+2]; v1 = src[j+3];
            h[v0]++; h[v1]++;
        }
        #endif
        for(; j < size.width; j++)
            h[src[j]]++;
    }

    int left_bound = 0, right_bound = 0, max_ind = 0, max = 0;
    int temp;
    bool isflipped = false;

    for(i = 0; i < N; i++)
    {
        if(h[i] > 0)
        {
            left_bound = i;
            break;
        }
    }
    if(left_bound > 0 )
        left_bound--;

    for(i = N-1; i > 0; i--)
    {
        if(h[i] > 0)
        {
            right_bound = i;
            break;
        }
    }
    if(right_bound < N-1)
        right_bound++;

    for(i = 0; i < N; i++)
    {
        if(h[i] > max)
        {
            max = h[i];
            max_ind = i;
        }
    }

    if(max_ind-left_bound < right_bound-max_ind)
    {
        isflipped = true;
        i = 0, j = N-1;
        while( i < j )
        {
            temp = h[i]; h[i] = h[j]; h[j] = temp;
            i++; j--;
        }
        left_bound = N-1-right_bound;
        max_ind = N-1-max_ind;
    }

    double thresh = left_bound;
    double a, b, dist = 0, tempdist;

    /*
     * We do not need to compute precise distance here. Distance is maximized, so some constants can
     * be omitted. This speeds up a computation a bit.
     */

    a = max; b = left_bound-max_ind;
    for(i = left_bound+1; i <= max_ind; i++)
    {
        tempdist = a*i + b*h[i];
        if(tempdist > dist)
        {
            dist = tempdist;
            thresh = i;
        }
    }
    thresh--;

    if(isflipped)
        thresh = N-1-thresh;

    return thresh;
}

static double getThreshVal_Otsu_8u(Mat src)
{
    int size[] = {src.rows, src.cols};
    int step = src.step[0];
    if(isContinuous(src))
    {
        size[1] *= size[0];
        size[0] = 1;
        step = size[1];
    }

    const int N = 256;
    int i, j, h[N] = {0};
    for(i = 0; i < size[0]; i++)
    {
        const unsigned char* _src = ptr(src, 0) + step*i;
        j = 0;
        #if CV_ENABLE_UNROLLED
        for(; j <= size[1]-4; j += 4)
        {
            int v0 = _src[j], v1 = _src[j+1];
            h[v0]++; h[v1]++;
            v0 = _src[j+2]; v1 = _src[j+3];
            h[v0]++; h[v1]++;
        }
        #endif
        for(;j < size[1];j++)
            h[_src[j]]++;
    }
    
    double mu = 0, scale = 1./(size.width*size.height);
    for(i = 0; i < N; i++)
        mu += i*(double)h[i];

    mu *= scale;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    for(i = 0; i < N; i++)
    {
        double p_i, q2, mu2, sigma;

        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if(min(q1,q2) < FLT_EPSILON || max(q1,q2) > 1.-FLT_EPSILON)
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        if(sigma > max_sigma)
        {
            max_sigma = sigma;
            max_val = i;
        }
    }

    return max_val;
}

double threshold(Mat* src, Mat* dst, double thresh, double maxval, int type)
{
    int automatic_thresh = (type & ~CV_THRESH_MASK);
    type &= THRESH_MASK;

    assert(automatic_thresh != (CV_THRESH_OTSU | CV_THRESH_TRIANGLE));
    if(automatic_thresh == CV_THRESH_OTSU)
    {
        CV_Assert(type(*src) == CV_8UC1);
        thresh = getThreshVal_Otsu_8u(*src);
    }
    else if(automatic_thresh == CV_THRESH_TRIANGLE)
    {
        assert(type(*src) == CV_8UC1);
        thresh = getThreshVal_Triangle_8u(*src);
    }
    create(dst, src->rows, src->cols, type(src));
    if(depth(*src) == CV_8U)
    {
        int ithresh = floor(thresh);
        thresh = ithresh;
        int imaxval = round(maxval);
        if(type == THRESH_TRUNC)
            imaxval = ithresh;
        imaxval = (unsigned char)imaxval;
        if(ithresh < 0 || ithresh >= 255)
        {
            if(type == THRESH_BINARY || type == THRESH_BINARY_INV ||
                ((type == THRESH_TRUNC || type == THRESH_TOZERO_INV) && ithresh < 0) ||
                (type == THRESH_TOZERO && ithresh >= 255))
            {
                int v = type == THRESH_BINARY ? (ithresh >= 255 ? 0 : imaxval) :
                        type == THRESH_BINARY_INV ? (ithresh >= 255 ? imaxval : 0) :
                        /*type == THRESH_TRUNC ? imaxval :*/ 0;

                setTo(dst, v);
            }
            else
                copyTo(*src, dst);
            return thresh;
        }

        thresh = ithresh;
        maxval = imaxval;
    }
    parallel_for_(init_Point(0, dst.rows) /* Interval */,
                  thresholdRunner(src, dst, thresh, maxval, type),
                  total(*dst)/(double)(1<<16));
}

double cvThreshold(const Mat* src, Mat* dst, double thresh, double maxval, int type)
{
    Mat dst0 = *dst;

    assert(src->rows == dst->rows && src->cols == dst->cols && channels(*src) == channels(*dst) &&
        (depth(*src) == depth(*dst) || depth(dst0) == CV_8U));

    thresh = threshold(src, dst, thresh, maxval, type);
    if(dst0.data != dst.data)
        convertTo(dst, dst0, depth(dst0), 1, 0);
    return thresh;
}

void cvInsertNodeIntoTree( Seq* _node, Seq* _parent, Seq* _frame)
{
    TreeNode* node = (TreeNode*)_node;
    TreeNode* parent = (TreeNode*)_parent;

    if( !node || !parent )
        fatal("NULL Pointer error");

    node->v_prev = _parent != _frame ? parent : 0;
    node->h_next = parent->v_next;

    assert(parent->v_next != node);

    if(parent->v_next)
        parent->v_next->h_prev = node;
    parent->v_next = node;
}

static void icvEndProcessContour(ContourScanner* scanner)
{
    ContourInfo* l_cinfo = scanner->l_cinfo;

    if(l_cinfo)
    {
        if(scanner->subst_flag)
        {
            MemStoragePos temp;

            SaveMemStoragePos(scanner->storage2, &temp);

            if(temp.top == scanner->backup_pos2.top &&
                temp.free_space == scanner->backup_pos2.free_space)
            {
                RestoreMemStoragePos(scanner->storage2, &scanner->backup_pos);
            }
            scanner->subst_flag = 0;
        }

        if(l_cinfo->contour)
        {
            cvInsertNodeIntoTree(l_cinfo->contour, l_cinfo->parent->contour,
                                  &(scanner->frame));
        }
        scanner->l_cinfo = 0;
    }
}

Seq* cvFindNextContour(ContourScanner* scanner)
{
    if(!scanner)
        fatal("NULL Point error");

    assert(scanner->img_step >= 0);

    icvEndProcessContour(scanner);

    /* initialize local state */
    signed char* img0 = scanner->img0;
    signed char* img = scanner->img;
    int step = scanner->img_step;
    int step_i = step / sizeof(int);
    int x = scanner->pt.x;
    int y = scanner->pt.y;
    int width = scanner->img_size.x;
    int height = scanner->img_size.y;
    int mode = scanner->mode;
    Point lnbd = scanner->lnbd;
    int nbd = scanner->nbd;
    int prev = img[x - 1];
    int new_mask = -2;

    if( mode == CV_RETR_FLOODFILL )
    {
        prev = ((int*)img)[x - 1];
        new_mask = INT_MIN / 2;
    }

    for(; y < height; y++, img += step)
    {
        int* img0_i = 0;
        int* img_i = 0;
        int p = 0;

        if(mode == CV_RETR_FLOODFILL)
        {
            img0_i = (int*)img0;
            img_i = (int*)img;
        }

        for(; x < width; x++)
        {
            if(img_i)
            {
                for(; x < width && ((p = img_i[x]) == prev || (p & ~new_mask) == (prev & ~new_mask)); x++ )
                    prev = p;
            }
            else
            {
                for(; x < width && (p = img[x]) == prev; x++)
                    ;
            }

            if(x >= width)
                break;


        }

    }
}

static ContourScanner* cvStartFindContours_Impl(Mat* mat, MemStorage* storage,
                     int  header_size, int mode,
                     int  method, Point offset, int needFillBorder)
{
    if(!storage)
        Fatal("NULL Pointer");

    if(!CV_IS_MASK_ARR(mat))
    {
        Fatal("[Start]FindContours supports only CV_8UC1 images when mode != CV_RETR_FLOODFILL ");
    }

    Point size = init_Point(mat->width, mat->height);
    int step = mat->step[0];
    unsigned char* img = (unsigned char*)(mat->data);
    
    if(header_size < sizeof(Contour))
        Fatal("Bad size error");

    ContourScanner* scanner = malloc(sizeof(*scanner));
    memset(scanner, 0, sizeof(*scanner));

    scanner->storage1 = scanner->storage2 = storage;
    scanner->img0 = (signed char*)img;
    scanner->img = (signed char*)(img + step);
    scanner->img_step = step;
    scanner->img_size.x = size.x - 1;   /* exclude rightest column */
    scanner->img_size.y = size.y - 1; /* exclude bottomost row */
    scanner->mode = mode;
    scanner->offset = offset;
    scanner->pt.x = scanner->pt.y = 1;
    scanner->lnbd.x = 0;
    scanner->lnbd.y = 1;
    scanner->nbd = 2;
    scanner->frame_info.contour = &(scanner->frame);
    scanner->frame_info.is_hole = 1;
    scanner->frame_info.next = 0;
    scanner->frame_info.parent = 0;
    scanner->frame_info.rect = init_Rect(0, 0, size.x, size.y);
    scanner->l_cinfo = 0;
    scanner->subst_flag = 0;

    scanner->frame.flags = CV_SEQ_FLAG_HOLE;

    scanner->approx_method2 = scanner->approx_method1 = method;

    if(method == CV_CHAIN_APPROX_TC89_L1 || method == CV_CHAIN_APPROX_TC89_KCOS)
        scanner->approx_method1 = CV_CHAIN_CODE;

    if(scanner->approx_method1 == CV_CHAIN_CODE)
    {
        scanner->seq_type1 = CV_SEQ_CHAIN_CONTOUR;
        scanner->header_size1 = scanner->approx_method1 == scanner->approx_method2 ?
            header_size : sizeof(Chain);
        scanner->elem_size1 = sizeof(char);
    }
    else
    {
        scanner->seq_type1 = CV_SEQ_POLYGON;
        scanner->header_size1 = scanner->approx_method1 == scanner->approx_method2 ?
            header_size : sizeof(Contour);
        scanner->elem_size1 = sizeof(Point);
    }

    scanner->header_size2 = header_size;

    if( scanner->approx_method2 == CV_CHAIN_CODE )
    {
        scanner->seq_type2 = scanner->seq_type1;
        scanner->elem_size2 = scanner->elem_size1;
    }
    else
    {
        scanner->seq_type2 = CV_SEQ_POLYGON;
        scanner->elem_size2 = sizeof(Point);
    }

    scanner->seq_type1 = scanner->approx_method1 == CV_CHAIN_CODE ?
        CV_SEQ_CHAIN_CONTOUR : CV_SEQ_POLYGON;

    scanner->seq_type2 = scanner->approx_method2 == CV_CHAIN_CODE ?
        CV_SEQ_CHAIN_CONTOUR : CV_SEQ_POLYGON;

    SaveMemStoragePos(storage, &(scanner->initial_pos));

    if(method > CV_CHAIN_APPROX_SIMPLE)
        scanner->storage1 = CreateChildMemStorage(scanner->storage2);

    if(mode > CV_RETR_LIST) 
    {
        scanner->cinfo_storage = CreateChildMemStorage(scanner->storage2);
        scanner->cinfo_set = CreateSet(0, sizeof(Set), sizeof(ContourInfo),
                                          scanner->cinfo_storage);
    }

    assert(step >= 0);
    assert(size.y >= 1);

    /* make zero borders */
    if(needFillBorder)
    {
        int esz = CV_ELEM_SIZE(mat->type);
        memset(img, 0, size.width*esz);
        memset(img + (size_t)(step)*(size.y-1), 0, size.x*esz);

        img += step;
        for(int y = 1; y < size.y-1; y++, img += step)
        {
            for(int k = 0; k < esz; k++)
                img[k] = img[(size.width-1)*esz+k] = (signed char)0;
        }
    }

    if(CV_MAT_TYPE(mat->type) != CV_32S)
        cvThreshold(mat, mat, 0, 1, CV_THRESH_BINARY);

    return scanner;
}

static int cvFindContours_Impl(Mat*  img,  MemStorage*  storage,
                Seq**  firstContour, int  cntHeaderSize,
                int  mode,
                int  method, CvPoint offset, int needFillBorder)
{
    ContourScanner* scanner = 0;
    Seq *contour = 0;
    int count = -1;

    if(!firstContour)
        Fatal("NULL double CvSeq pointer");

    *firstContour = 0;
    scanner = cvStartFindContours_Impl(img, storage, cntHeaderSize, mode, method, offset,
                                            needFillBorder);

}

void findContours(Mat image0, vector** contours, vector* hierarchy, int mode, int method, Point offset)
{
    Point offset0;
    offset0.x = offset0.y = -1;
    Scalar s = init_Scalar(0, 0, 0, 0);
    Mat* image;
    copyMakeBorder(image0, image, 1, 1, 1, 1, BORDER_CONSTANT | BORDER_ISOLATED, s);
    MemStorage* storage = malloc(sizeof(MemStorage));
    init_MemStorage(storage, 0);
    Seq* _ccontours = 0;
    cvFindContours_Impl(image, storage, &_ccontours, sizeof(Contour), mode, method, init_Point(offset.x + offset0.x, offset.y + offset0.y), 0);
}

void setCallback(ERFilterNM* filter,ERClassifierNM *erc)
{
	filter->classifier = *erc;
}

void setMinArea(ERFilterNM* filter, float _minArea)
{
	filter->minArea = _minArea;
}

void setMaxArea(ERFilterNM* filter, float _maxArea)
{
	filter->maxArea = _maxArea;
}

void setMinProbability(ERFilterNM* filter, float _minProbability)
{
	filter->minProbability = _minProbability;
}

void setNonMaxSuppression(ERFilterNM* filter, bool _nonMaxSuppression)
{
	filter->nonMaxSuppression = _nonMaxSuppression;
}

void setMinProbabilityDiff(ERFilterNM* filter, float _minProbabilityDiff)
{
	filter->minProbabilityDiff = _minProbabilityDiff;
}

