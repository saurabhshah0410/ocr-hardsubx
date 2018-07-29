#include <stdio.h>

#define CV_DTREE_CAT_DIR(idx,subset) \
        (2*((subset[(idx)>>5]&(1 << ((idx) & 31)))==0)-1)
#define CV_NEXT_SEQ_ELEM(elem_size, reader)                 \
{                                                             \
    if( ((reader).ptr += (elem_size)) >= (reader).block_max ) \
    {                                                         \
        ChangeSeqBlock( &(reader), 1 );                     \
    }                                                         \
}
#define CV_NODE_IS_INT(flags)       (CV_NODE_TYPE(flags) == CV_NODE_INT)
#define CV_NODE_IS_REAL(flags)       (CV_NODE_TYPE(flags) == CV_NODE_REAL)
#define CV_NODE_IS_MAP(flags)        (CV_NODE_TYPE(flags) == CV_NODE_MAP)
#define CV_NODE_IS_COLLECTION(flags) (CV_NODE_TYPE(flags) >= CV_NODE_SEQ)
#define cvFree(ptr) (cvFree_(*(ptr)), *(ptr)=0)
#define CV_NODE_TYPE(flags)  ((flags) & CV_NODE_TYPE_MASK)

#define CV_NODE_IS_USER(flags)       (((flags) & CV_NODE_USER) != 0)

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

#define cv_isprint(c)     ((uchar)(c) >= (uchar)' ')

#define CV_GET_LAST_ELEM( seq, block ) \
    ((block)->data + ((block)->count - 1)*((seq)->elem_size))

#define CV_READ_CHAIN_POINT( _pt, reader )                              \
{                                                                       \
    (_pt) = (reader).pt;                                                \
    if( (reader).ptr )                                                  \
    {                                                                   \
        CV_READ_SEQ_ELEM( (reader).code, (reader));                     \
        assert( ((reader).code & ~7) == 0 );                            \
        (reader).pt.x += (reader).deltas[(int)(reader).code][0];        \
        (reader).pt.y += (reader).deltas[(int)(reader).code][1];        \
    }                                                                   \
}

#define CV_SEQ_KIND(seq)     ((seq)->flags & CV_SEQ_KIND_MASK)

#define CV_IS_SEQ_CHAIN(seq)   \
    (CV_SEQ_KIND(seq) == CV_SEQ_KIND_CURVE && (seq)->elem_size == 1)

#define CV_IS_SEQ_CLOSED(seq)     (((seq)->flags & CV_SEQ_FLAG_CLOSED) != 0)

#define CV_IS_SEQ_CHAIN_CONTOUR(seq) \
    (CV_IS_SEQ_CHAIN(seq) && CV_IS_SEQ_CLOSED(seq))

static const Point icvCodeDeltas[8] =
    { init_Point(1, 0), init_Point(1, -1), init_Point(0, -1), init_Point(-1, -1), init_Point(-1, 0), init_Point(-1, 1), init_Point(0, 1), init_Point(1, 1) };

#define  CV_TOGGLE_FLT(x) ((x)^((int)(x) < 0 ? 0x7fffffff : 0))

#define CV_SEQ_ELTYPE(seq)   ((seq)->flags & CV_SEQ_ELTYPE_MASK)

/** type checking macro */
#define CV_IS_SEQ_POINT_SET(seq) \
    ((CV_SEQ_ELTYPE(seq) == CV_32SC2 || CV_SEQ_ELTYPE(seq) == CV_32FC2))

#define CV_IS_SEQ(seq) \
    ((seq) != NULL && (((CvSeq*)(seq))->flags & CV_MAGIC_MASK) == CV_SEQ_MAGIC_VAL)

#define CV_WRITE_SEQ_ELEM(elem, writer)             \
{                                                     \
    assert((writer).seq->elem_size == sizeof(elem)); \
    if((writer).ptr >= (writer).block_max)          \
    {                                                 \
        CreateSeqBlock(&writer);                   \
    }                                                 \
    assert((writer).ptr <= (writer).block_max - sizeof(elem));\
    memcpy((writer).ptr, &(elem), sizeof(elem));      \
    (writer).ptr += sizeof(elem);                     \
}

#define CV_IS_SEQ_HOLE(seq)       (((seq)->flags & CV_SEQ_FLAG_HOLE) != 0)

#define CV_SEQ_READER_FIELDS()                                      \
    int          header_size;                                       \
    Seq*       seq;        /**< sequence, beign read */             \
    SeqBlock*  block;      /**< current block */                    \
    signed char*       ptr;        /**< pointer to element be read next */  \
    signed char*       block_min;  /**< pointer to the beginning of block */\
    signed char*       block_max;  /**< pointer to the end of block */      \
    int          delta_index;/**< = seq->first->start_index   */      \
    signed char*       prev_elem;  /**< pointer to previous element */

#define CV_SEQ_WRITER_FIELDS()                                     \
    int          header_size;                                      \
    Seq*       seq;        /**< the sequence written */            \
    SeqBlock*  block;      /**< current block */                   \
    signed char*       ptr;        /**< pointer to free space */           \
    signed char*       block_min;  /**< pointer to the beginning of block*/\
    signed char*       block_max;  /**< pointer to the end of block */

/* initializes 8-element array for fast access to 3x3 neighborhood of a pixel */
#define  CV_INIT_3X3_DELTAS(deltas, step, nch)              \
    ((deltas)[0] =  (nch),  (deltas)[1] = -(step) + (nch),  \
     (deltas)[2] = -(step), (deltas)[3] = -(step) - (nch),  \
     (deltas)[4] = -(nch),  (deltas)[5] =  (step) - (nch),  \
     (deltas)[6] =  (step), (deltas)[7] =  (step) + (nch))

#define CV_TREE_NODE_FIELDS(node_type)                                \
    int flags;                  /**< Miscellaneous flags.     */      \
    int header_size;            /**< Size of sequence header. */      \
    struct node_type* h_prev;   /**< Previous sequence.       */      \
    struct node_type* h_next;   /**< Next sequence.           */      \
    struct node_type* v_prev;   /**< 2nd previous sequence.   */      \
    struct node_type* v_next    /**< 2nd next sequence.       */

#define CV_SEQUENCE_FIELDS()                                                      \
    CV_TREE_NODE_FIELDS(Seq);                                                     \
    int total;              /**< Total number of elements.            */          \
    int elem_size;          /**< Size of sequence element in bytes.   */          \
    signed char* block_max; /**< Maximal bound of the last block.     */          \
    signed char* ptr;       /**< Current write pointer.               */          \
    int delta_elems;        /**< Grow seq this many at a time.        */          \
    MemStorage* storage;    /**< Where the seq is stored.             */          \
    SeqBlock* free_blocks;  /**< Free blocks list.                    */          \
    SeqBlock* first;        /**< Pointer to the first sequence block. */

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
    unsigned char** ptrs;
    //! the number of arrays
    int narrays;
    //! the number of hyper-planes that the iterator steps through
    size_t nplanes;
    //! the size of each segment (in elements)
    size_t size;

    int iterdepth;
    size_t idx;
} MatIterator ;

typedef struct MatConstIterator
{
    const Mat* m;
    size_t elemSize;
    const unsigned char* ptr;
    const unsigned char* sliceStart;
    const unsigned char* sliceEnd;
} MatConstIterator ;

void seek(MatConstIterator* it, ptrdiff_t ofs, bool relative)
{
    if(isContinuous(&(it->m)))
    {
        if(it->ptr < it->sliceStart)
            it->ptr = it->sliceStart;
        else if(it->ptr > it->sliceEnd)
            it->ptr = it->sliceEnd;
        return;
    }

    ptrdiff_t ofs0, y;
    if(relative)
    {
        ofs0 = it->ptr - ptr(*(it->m), 0);
        y = ofs0/it->m->step[0];
        ofs += y*it->m->cols + (ofs0 - y*it->m->step[0])/elemSize;
    }

    y = ofs/m->cols;
    int y1 = min(max((int)y, 0), it->m->rows-1);
    it->sliceStart = ptr(*(it->m), y1);
    it->sliceEnd = it->sliceStart + it->m->cols*it->elemSize;
    it->ptr = y < 0 ? it->sliceStart : y >= it->m->rows ? it->sliceEnd :
    it->sliceStart + (ofs - y*it->m->cols)*it->elemSize;
    return;
}

void leftshift_op(Mat* m, int cols, float sample[])
{
    MatConstIterator it;
    it.m = malloc(sizeof(Mat));
    it.ptr = malloc(sizeof(unsigned char));
    it.sliceStart = malloc(sizeof(unsigned char));
    it.sliceEnd = malloc(sizeof(unsigned char));
    it.elemSize = elemSize(*m);
    it.m = m;
    it.ptr = 0;
    it.sliceStart = 0;
    it.sliceEnd = 0;


    if(isContinuous(*m))
    {
        it.sliceStart = ptr(m, 0);
        it.sliceEnd = it.sliceStart + total(*m)*it.elemSize;
    }
    seek(&it, 0, false);

    for(int i = 0; i < cols; i++)
    {
        *(float*)it.ptr = float(sample[i]);
        if(it.m && (it.ptr += it.elemSize) >= it.sliceEnd)
        {
            it.ptr -= it.elemSize;
            seek(&it, 1, true);
        }
    }
}

typedef struct RGBtoGray
{
    int srccn;
    int tab[256*3];
} RGBtoGray ;

void RGB2Gray(RGBtoGray* r2g, int _srccn, int blueIdx, const int* coeffs)
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

typedef struct Size
{
    int width;
    int height;
} Size ;

typedef struct CvtHelper
{
    Mat src, dst;
    int depth, scn;
    Size dstSz;
} CvtHelper ;

// Represents a pair of ER's
typedef struct region_pair
{
    Point a;
    Point b;
} region_pair ;

// struct region_sequence
// Represents a sequence of more than three ER's
typedef struct region_sequence
{
    vector* triplets; //region_triplet
};

region_sequence Region_sequence(region_triplet* t)
{
    region_sequence sequence;
    sequence.triplets = malloc(sizeof(vector));
    vector_add(sequence.triplets, t);
    return sequence;
}

// struct line_estimates
// Represents a line estimate (as above) for an ER's group
// i.e.: slope and intercept of 2 top and 2 bottom lines
typedef struct line_estimates
{
    float top1_a0;
    float top1_a1;
    float top2_a0;
    float top2_a1;
    float bottom1_a0;
    float bottom1_a1;
    float bottom2_a0;
    float bottom2_a1;
    int x_min;
    int x_max;
    int h_max;
} line_estimates ;


bool equal_line_estimates(line_estimates e1, line_estimates e2)
{
    return ((e1.top1_a0 == e2.top1_a0) && (e1.top1_a1 == e2.top1_a1) && (e1.top2_a0 == e2.top2_a0) &&
        (e1.top2_a1 == e2.top2_a1) && (e1.bottom1_a0 == e2.bottom1_a0) && (e1.bottom1_a1 == e2.bottom1_a1) &&
        (e1.bottom2_a0 == e2.bottom2_a0) && (e1.bottom2_a1 == e2.bottom2_a1) && (e1.x_min == e2.x_min) &&
        (e1.x_max == e2.x_max) && (e1.h_max == e2.h_max));
}

// struct region_triplet
// Represents a triplet of ER's
typedef struct region_triplet
{
    Point a;
    Point b;
    Point c;
    line_estimates estimates;
} region_triplet ;


bool equalRegionPairs(region_pair r1, region_pair r2)
{
    return r1.a.x == r2.a.x && r1.a.y == r2.a.y && r1.b.x == r2.b.x && r1.b.y == r2.b.y;
}

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

typedef struct SeqReader
{
    CV_SEQ_READER_FIELDS()
} SeqReader ;


typedef union Cv32suf
{
    int i;
    unsigned u;
    float f;
    struct _fp32Format
    {
        unsigned int significand : 23;
        unsigned int exponent    : 8;
        unsigned int sign        : 1;
    } fmt;
}
Cv32suf;


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
    int type; // type = 1 --> evalNM1, type = 2 --> evalNM2
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

/** Freeman chain reader state */
typedef struct ChainPtReader
{
    CV_SEQ_READER_FIELDS()
    char      code;
    Point   pt;
    signed char     deltas[8][2];
} ChainPtReader ;

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
    unsigned char* dst_data;
    const size_t dst_step;
    const int width;
    const void* cvt;
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

typedef struct SeqWriter
{
    CV_SEQ_WRITER_FIELDS()
} SeqWriter;

typedef struct PtInfo
{
    Point pt;
    int k;                      /* support region */
    int s;                      /* curvature value */
    struct PtInfo *next;
} PtInfo ;

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

typedef struct TreeNodeIterator
{
    const void* node;
    int level;
    int max_level;
} TreeNodeIterator ;

typedef struct CvString
{
    int len;
    char* ptr;
} CvString ;

typedef void (*CvStartWriteStruct)( struct CvFileStorage* fs, const char* key,
                                    int struct_flags, const char* type_name );
typedef void (*CvEndWriteStruct)( struct CvFileStorage* fs );
typedef void (*CvWriteInt)( struct CvFileStorage* fs, const char* key, int value );
typedef void (*CvWriteReal)( struct CvFileStorage* fs, const char* key, double value );
typedef void (*CvWriteString)( struct CvFileStorage* fs, const char* key,
                               const char* value, int quote );
typedef void (*CvWriteComment)( struct CvFileStorage* fs, const char* comment, int eol_comment );
typedef void (*CvStartNextStream)( struct CvFileStorage* fs );

typedef struct StringHash
{
    CV_SET_FIELDS()
    int tab_size;
    void** table;
} StringHash ;

typedef struct CvFileStorage
{
    int flags;
    int fmt;
    int write_mode;
    int is_first;
    MemStorage* memstorage;
    MemStorage* dststorage;
    MemStorage* strstorage;
    StringHash* str_hash;
    Seq* roots;
    Seq* write_stack;
    int struct_indent;
    int struct_flags;
    CvString struct_tag;
    int space;
    char* filename;
    FILE* file;
    void* gzfile;
    char* buffer;
    char* buffer_start;
    char* buffer_end;
    int wrap_margin;
    int lineno;
    int dummy_eof;
    const char* errmsg;
    char errmsgbuf[128];

    CvStartWriteStruct start_write_struct;
    CvEndWriteStruct end_write_struct;
    CvWriteInt write_int;
    CvWriteReal write_real;
    CvWriteString write_string;
    CvWriteComment write_comment;
    CvStartNextStream start_next_stream;

    const char* strbuf;
    size_t strbufsize, strbufpos;
    vector* outbuf; //char

    bool is_write_struct_delayed;
    char* delayed_struct_key;
    int   delayed_struct_flags;
    char* delayed_type_name;

    bool is_opened;
} CvFileStorage ;

typedef struct FileStorage
{
    CvFileStorage* fs;
    char* elname; //!< the currently written element
    vector* structs //!< the stack of written structures(char)
    int state; //!< the writer state
} FileStorage ;

typedef struct AutoBuffer
{
    size_t fixed_size;
    //! pointer to the real buffer, can point to buf if the buffer is small enough
    void* ptr;
    //! size of the real buffer
    size_t sz;
    //! pre-allocated buffer. At least 1 element to confirm C++ standard requirements
    void* buf;
} AutoBuffer ;

void* fastMalloc(size_t size)
{
    uchar* udata = (uchar*)malloc(size + sizeof(void*) + CV_MALLOC_ALIGN);
    if(!udata)
        fatal("Out of Memory Error");
    uchar** adata = alignPtr((uchar**)udata + 1, CV_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

void* cvAlloc(size_t size)
{
    return fastMalloc(size);
}

StringHash* CreateMap(int flags, int header_size, int elem_size, CvMemStorage* storage, int start_tab_size)
{
    if( header_size < (int)sizeof(StringHash) )
        fatal("Too small map header_size");

    if( start_tab_size <= 0 )
        start_tab_size = 16;

    StringHash* map = (StringHash*)CreateSet(flags, header_size, elem_size, storage);

    map->tab_size = start_tab_size;
    start_tab_size *= sizeof(map->table[0]);
    map->table = (void**)MemStorageAlloc( storage, start_tab_size );
    memset(map->table, 0, start_tab_size);

    return map;
}

/** All the keys (names) of elements in the readed file storage
   are stored in the hash to speed up the lookup operations: */
typedef struct StringHashNode
{
    unsigned hashval;
    char* str;
    struct StringHashNode* next;
} StringHashNode ;

typedef struct AttrList
{
    const char** attr;         /**< NULL-terminated array of (attribute_name,attribute_value) pairs. */
    struct AttrList* next;   /**< Pointer to next chunk of the attributes list.                    */
} AttrList ;

char* icvGets(CvFileStorage* fs, char* str, int maxCount)
{
    if( fs->file )
    {
        char* ptr = fgets( str, maxCount, fs->file );
        if (ptr && maxCount > 256 && !(fs->flags & 64))
        {
            size_t sz = strnlen(ptr, maxCount);
            assert(sz < (size_t)(maxCount - 1));
        }
        return ptr;
    }
}

static char* icvXMLSkipSpaces(CvFileStorage* fs, char* ptr, int mode)
{
    int level = 0;

    for(;;)
    {
        char c;
        ptr--;

        if(mode == CV_XML_INSIDE_TAG || mode == 0)
        {
            do c = *++ptr;
            while(c == ' ' || c == '\t');

            if(c == '<' && ptr[1] == '!' && ptr[2] == '-' && ptr[3] == '-')
            {
                if(mode != 0)
                    CV_PARSE_ERROR( "Comments are not allowed here" );
                mode = CV_XML_INSIDE_COMMENT;
                ptr += 4;
            }
            else if( cv_isprint(c) )
                break;
        }

        if(!cv_isprint(*ptr))
        {
            int max_size = (int)(fs->buffer_end - fs->buffer_start);

            ptr = icvGets(fs, fs->buffer_start, max_size);
            if(!ptr)
            {
                ptr = fs->buffer_start;  // FIXIT Why do we need this hack? What is about other parsers JSON/YAML?
                *ptr = '\0';
                fs->dummy_eof = 1;
                break;
            }
            else
                int l = (int)strlen(ptr);

            fs->lineno++;  // FIXIT doesn't really work with long lines. It must be counted via '\n' or '\r' symbols, not the number of icvGets() calls.
        }
    }
    return ptr;
}

inline bool cv_isalnum(char c)
{
    return ('0' <= c && c <= '9') || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

/** Fast variant of cvSetAdd */
inline SetElem* SetNew(Set* set_header)
{
    SetElem* elem = set_header->free_elems;
    if(elem)
    {
        set_header->free_elems = elem->next_free;
        elem->flags = elem->flags & CV_SET_ELEM_IDX_MASK;
        set_header->active_count++;
    }
    else
        cvSetAdd(set_header, NULL, &elem);
    return elem;
}

StringHashNode* GetHashedKey(CvFileStorage* fs, const char* str, int len, int create_missing)
{
    StringHashNode* node = 0;
    unsigned hashval = 0;
    int i, tab_size;

    if(!fs)
        return 0;

    StringHash* map = fs->str_hash;

    if(len < 0)
    {
        for( i = 0; str[i] != '\0'; i++ )
            hashval = hashval*CV_HASHVAL_SCALE + (unsigned char)str[i];
        len = i;
    }
    else for(i = 0; i < len; i++)
        hashval = hashval*CV_HASHVAL_SCALE + (unsigned char)str[i];

    hashval &= INT_MAX;
    tab_size = map->tab_size;
    if((tab_size & (tab_size - 1)) == 0)
        i = (int)(hashval & (tab_size - 1));
    else
        i = (int)(hashval % tab_size);

    for(node = (StringHashNode*)(map->table[i]); node != 0; node = node->next)
    {
        if(node->hashval == hashval &&
            node->str.len == len &&
            memcmp(node->str.ptr, str, len) == 0)
            break;
    }
    if(!node && create_missing)
    {
        node = (StringHashNode*)SetNew((Set*)map);
        node->hashval = hashval;
        node->str = cvMemStorageAllocString( map->storage, str, len );
        node->next = (CvStringHashNode*)(map->table[i]);
        map->table[i] = node;
    }

    return node;
}

typedef int (*IsInstanceFunc)(const void* struct_ptr);
typedef void (*ReleaseFunc)(void** struct_dblptr);
typedef void* (*ReadFunc)(CvFileStorage* storage, CvFileNode* node);
typedef void (*WriteFunc)(CvFileStorage* storage, const char* name,
                                      const void* struct_ptr, CvAttrList attributes);
typedef void* (*CloneFunc)(const void* struct_ptr);

typedef struct TypeInfo
{
    int flags; /**< not used */
    int header_size; /**< sizeof(TypeInfo) */
    struct TypeInfo* prev; /**< previous registered type in the list */
    struct TypeInfo* next; /**< next registered type in the list */
    const char* type_name; /**< type name, written to file storage */
    IsInstanceFunc is_instance; /**< checks if the passed object belongs to the type */
    ReleaseFunc release; /**< releases object (memory etc.) */
    ReadFunc read; /**< reads object from file storage */
    WriteFunc write; /**< writes object to file storage */
    CloneFunc clone; /**< creates a copy of the object */
} TypeInfo ;

/** Basic element of the file storage - scalar or collection: */
typedef struct CvFileNode
{
    int tag;
    struct TypeInfo* info; /**< type information
            (only for user-defined object, for others it is 0) */
    union
    {
        double f; /**< scalar floating-point number */
        int i;    /**< scalar integer number */
        char* str; /**< text string */
        Seq* seq; /**< sequence (ordered collection of file nodes) */
        StringHash* map; /**< map (collection of named file nodes) */
    } data;
} CvFileNode ;

typedef struct FileNode
{
    const CvFileStorage* fs;
    const CvFileNode* node;
} FileNode ;

inline bool cv_isspace(char c)
{
    return (9 <= c && c <= 13) || c == ' ';
}

const char* AttrValue(const AttrList* attr, const char* attr_name)
{
    while(attr && attr->attr)
    {
        int i;
        for(i = 0; attr->attr[i*2] != 0; i++)
        {
            if(strcmp( attr_name, attr->attr[i*2] ) == 0 )
                return attr->attr[i*2+1];
        }
        attr = attr->next;
    }

    return 0;
}

TypeInfo* FindType(const char* type_name)
{
    TypeInfo* info = 0;

    if (type_name)
      for(info = 0; info != 0; info = info->next)
        if(strcmp( info->type_name, type_name ) == 0)
      break;

    return info;
}

typedef struct FileMapNode
{
    CvFileNode value;
    const StringHashNode* key;
    struct FileMapNode* next;
} FileMapNode ;

void icvFSCreateCollection(CvFileStorage* fs, int tag, CvFileNode* collection)
{
    if(CV_NODE_IS_MAP(tag))
        collection->data.map = CreateMap(0, sizeof(StringHash), sizeof(FileMapNode), fs->memstorage, 16);
    else
    {
        Seq* seq;
        seq = CreateSeq(0, sizeof(Seq), sizeof(CvFileNode), fs->memstorage);

        // if <collection> contains some scalar element, add it to the newly created collection
        if(CV_NODE_TYPE(collection->tag) != CV_NODE_NONE)
            SeqPush(seq, collection);

        collection->data.seq = seq;
    }

    collection->tag = tag;
    SetSeqBlockSize(collection->data.seq, 8);
}

inline bool cv_isdigit(char c)
{
    return '0' <= c && c <= '9';
}

inline bool cv_isalpha(char c)
{
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
}

static void icvProcessSpecialDouble(CvFileStorage* fs, char* buf, double* value, char** endptr)
{
    char c = buf[0];
    int inf_hi = 0x7ff00000;

    if(c == '-' || c == '+')
    {
        inf_hi = c == '-' ? 0xfff00000 : 0x7ff00000;
        c = *++buf;
    }

    union{double d; uint64 i;} v;
    v.d = 0.;
    if(toupper(buf[1]) == 'I' && toupper(buf[2]) == 'N' && toupper(buf[3]) == 'F')
        v.i = (uint64)inf_hi << 32;
    else if(toupper(buf[1]) == 'N' && toupper(buf[2]) == 'A' && toupper(buf[3]) == 'N')
        v.i = (uint64)-1;

    *value = v.d;
    *endptr = buf + 4;
}

double icv_strtod(CvFileStorage* fs, char* ptr, char** endptr)
{
    double fval = strtod(ptr, endptr);
    if(**endptr == '.')
    {
        char* dot_pos = *endptr;
        *dot_pos = ',';
        double fval2 = strtod(ptr, endptr);
        *dot_pos = '.';
        if(*endptr > dot_pos)
            fval = fval2;
        else
            *endptr = dot_pos;
    }

    if(*endptr == ptr || cv_isalpha(**endptr))
        icvProcessSpecialDouble( fs, ptr, &fval, endptr );

    return fval;
}

/* Find a sequence element by its index: */
signed char* GetSeqElem(const Seq *seq, int index)
{
    SeqBlock *block;
    int count, total = seq->total;

    if((unsigned)index >= (unsigned)total)
    {
        index += index < 0 ? total : 0;
        index -= index >= total ? total : 0;
        if((unsigned)index >= (unsigned)total)
            return 0;
    }

    block = seq->first;
    if(index + index <= total)
    {
        while(index >= (count = block->count))
        {
            block = block->next;
            index -= count;
        }
    }
    else
    {
        do
        {
            block = block->prev;
            total -= block->count;
        }
        while(index < total);
        index -= total;
    }

    return block->data + index * seq->elem_size;
}

CvFileNode* cvGetFileNode(CvFileStorage* fs, CvFileNode* _map_node, const StringHashNode* key, int create_missing)
{
    CvFileNode* value = 0;
    int k = 0, attempts = 1;

    if(!fs)
        return 0;

    if(_map_node)
    {
        if(!fs->roots)
            return 0;
        attempts = fs->roots->total;
    }

    for(k = 0; k < attempts; k++)
    {
        int i, tab_size;
        CvFileNode* map_node = _map_node;
        FileMapNode* another;
        StringHash* map;

        if(!map_node)
            map_node = (CvFileNode*)GetSeqElem(fs->roots, k);
        if(!CV_NODE_IS_MAP(map_node->tag))
        {
            return 0;
        }

        map = map_node->data.map;
        tab_size = map->tab_size;

        if( (tab_size & (tab_size - 1)) == 0 )
            i = (int)(key->hashval & (tab_size - 1));
        else
            i = (int)(key->hashval % tab_size);

        for( another = (FileMapNode*)(map->table[i]); another != 0; another = another->next )
            if( another->key == key )
            {
                if(!create_missing)
                {
                    value = &another->value;
                    return value;
                }
            }

        if(k == attempts - 1 && create_missing)
        {
            FileMapNode* node = (FileMapNode*)SetNew((Set*)map);
            node->key = key;

            node->next = (FileMapNode*)(map->table[i]);
            map->table[i] = node;
            value = (CvFileNode*)node;
        }
    }

    return value;
}

static char* icvXMLParseValue(CvFileStorage* fs, char* ptr, CvFileNode* node, int value_type)
{
    CvFileNode *elem = node;
    bool have_space = true, is_simple = true;
    int is_user_type = CV_NODE_IS_USER(value_type);
    memset(node, 0, sizeof(*node));

    value_type = CV_NODE_TYPE(value_type);

    for(;;)
    {
        char c = *ptr, d;
        char* endptr;

        if(cv_isspace(c) || c == '\0' || (c == '<' && ptr[1] == '!' && ptr[2] == '-'))
        {
            ptr = icvXMLSkipSpaces(fs, ptr, 0);
            have_space = true;
            c = *ptr;
        }

        d = ptr[1];

        if( c =='<' || c == '\0' )
        {
            StringHashNode *key = 0, *key2 = 0;
            AttrList* list = 0;
            TypeInfo* info = 0;
            int tag_type = 0;
            int is_noname = 0;
            const char* type_name = 0;
            int elem_type = CV_NODE_NONE;

            if( d == '/' || c == '\0' )
                break;

            ptr = icvXMLParseTag(fs, ptr, &key, &list, &tag_type);

            /* for base64 string */
            bool is_binary_string = false;

            type_name = list ? AttrValue(list, "type_id") : 0;
            if(type_name)
            {
                if(strcmp( type_name, "str") == 0)
                    elem_type = CV_NODE_STRING;
                else if(strcmp(type_name, "map") == 0)
                    elem_type = CV_NODE_MAP;
                else if(strcmp(type_name, "seq") == 0)
                    elem_type = CV_NODE_SEQ;
                else if (strcmp(type_name, "binary") == 0)
                {
                    elem_type = CV_NODE_NONE;
                    is_binary_string = true;
                }
                else
                {
                    info = FindType(type_name);
                    if(info)
                        elem_type = CV_NODE_USER;
                }
            }

            is_noname = key->str.len == 1 && key->str.ptr[0] == '_';
            if(!CV_NODE_IS_COLLECTION(node->tag))
            {
                icvFSCreateCollection(fs, is_noname ? CV_NODE_SEQ : CV_NODE_MAP, node);
            }

            if(is_noname)
                elem = (CvFileNode*)SeqPush( node->data.seq, 0 );
            else
                elem = cvGetFileNode(fs, node, key, 1);
            assert(elem);
            if (!is_binary_string)
                ptr = icvXMLParseValue(fs, ptr, elem, elem_type);
            
            if(!is_noname)
                elem->tag |= CV_NODE_NAMED;
            is_simple = is_simple && !CV_NODE_IS_COLLECTION(elem->tag);
            elem->info = info;
            ptr = icvXMLParseTag( fs, ptr, &key2, &list, &tag_type );
            have_space = true;
        }
        else
        {
            elem = node;
            if(node->tag != CV_NODE_NONE)
            {
                if(!CV_NODE_IS_COLLECTION(node->tag))
                    icvFSCreateCollection(fs, CV_NODE_SEQ, node);

                elem = (CvFileNode*)SeqPush(node->data.seq, 0);
                elem->info = 0;
            }

            if(value_type != CV_NODE_STRING &&
                (cv_isdigit(c) || ((c == '-' || c == '+') &&
                (cv_isdigit(d) || d == '.')) || (c == '.' && cv_isalnum(d)))) // a number
            {
                double fval;
                int ival;
                endptr = ptr + (c == '-' || c == '+');
                while(cv_isdigit(*endptr))
                    endptr++;
                if(*endptr == '.' || *endptr == 'e')
                {
                    fval = icv_strtod(fs, ptr, &endptr);
                    elem->tag = CV_NODE_REAL;
                    elem->data.f = fval;
                }
                else
                {
                    ival = (int)strtol( ptr, &endptr, 0 );
                    elem->tag = CV_NODE_INT;
                    elem->data.i = ival;
                }

                ptr = endptr;
            }
            else
            {
                //string
                char buf[CV_FS_MAX_LEN+16] = {0};
                int i = 0, len, is_quoted = 0;
                elem->tag = CV_NODE_STRING;
                if( c == '\"' )
                    is_quoted = 1;
                else
                    --ptr;

                for( ;; )
                {
                    c = *++ptr;
                    if(!cv_isalnum(c))
                    {
                        if( c == '\"' )
                        {
                            ++ptr;
                            break;
                        }
                        else if(!cv_isprint(c) || c == '<' || (!is_quoted && cv_isspace(c)))
                            break;

                        else if( c == '&' )
                        {
                            if( *++ptr == '#' )
                            {
                                int val, base = 10;
                                ptr++;
                                if( *ptr == 'x' )
                                {
                                    base = 16;
                                    ptr++;
                                }
                                val = (int)strtol( ptr, &endptr, base );
                                c = (char)val;
                            }
                            else
                            {
                                endptr = ptr;
                                do c = *++endptr;
                                while( cv_isalnum(c) );
                                len = (int)(endptr - ptr);
                                if( len == 2 && memcmp( ptr, "lt", len ) == 0 )
                                    c = '<';
                                else if( len == 2 && memcmp( ptr, "gt", len ) == 0 )
                                    c = '>';
                                else if( len == 3 && memcmp( ptr, "amp", len ) == 0 )
                                    c = '&';
                                else if( len == 4 && memcmp( ptr, "apos", len ) == 0 )
                                    c = '\'';
                                else if( len == 4 && memcmp( ptr, "quot", len ) == 0 )
                                    c = '\"';
                                else
                                {
                                    memcpy( buf + i, ptr-1, len + 2 );
                                    i += len + 2;
                                }
                            }
                            ptr = endptr;
                        }
                    }
                    buf[i++] = c;
                }
                elem->data.str = cvMemStorageAllocString(fs->memstorage, buf, i);
            }

            if(!CV_NODE_IS_COLLECTION(value_type) && value_type != CV_NODE_NONE)
                break;
            have_space = false;
        }
    }

    if((CV_NODE_TYPE(node->tag) == CV_NODE_NONE ||
        (CV_NODE_TYPE(node->tag) != value_type &&
        !CV_NODE_IS_COLLECTION(node->tag))) &&
        CV_NODE_IS_COLLECTION(value_type))
    {
        icvFSCreateCollection(fs, CV_NODE_IS_MAP(value_type) ?
                                        CV_NODE_MAP : CV_NODE_SEQ, node);
    }

    if( CV_NODE_IS_COLLECTION(node->tag) && is_simple )
        node->data.seq->flags |= CV_NODE_SEQ_SIMPLE;

    node->tag |= is_user_type ? CV_NODE_USER : 0;
    return ptr;
}

static char* icvXMLParseTag(CvFileStorage* fs, char* ptr, StringHashNode** _tag, AttrList** _list, int* _tag_type)
{
    int tag_type = 0;
    StringHashNode* tagname = 0;
    AttrList *first = 0, *last = 0;
    int count = 0, max_count = 4;
    int attr_buf_size = (max_count*2 + 1)*sizeof(char*) + sizeof(AttrList);
    char* endptr;
    char c;
    int have_space;

    ptr++;
    if(cv_isalnum(*ptr) || *ptr == '_')
        tag_type = CV_XML_OPENING_TAG;
    else if(*ptr == '/')
    {
        tag_type = CV_XML_CLOSING_TAG;
        ptr++;
    }
    else if(*ptr == '?')
    {
        tag_type = CV_XML_HEADER_TAG;
        ptr++;
    }
    else if(*ptr == '!')
    {
        tag_type = CV_XML_DIRECTIVE_TAG;
        assert( ptr[1] != '-' || ptr[2] != '-' );
        ptr++;
    }

    for(;;)
    {
        StringHashNode* attrname;

        endptr = ptr - 1;
        do c = *++endptr;
        while(cv_isalnum(c) || c == '_' || c == '-');

        attrname = GetHashedKey(fs, ptr, (int)(endptr - ptr), 1);
        assert(attrname);
        ptr = endptr;

        if(!tagname)
            tagname = attrname;
        else
        {
            if(!last || count >= max_count)
            {
                AttrList* chunk;

                chunk = (AttrList*)MemStorageAlloc(fs->memstorage, attr_buf_size);
                memset(chunk, 0, attr_buf_size);
                chunk->attr = (const char**)(chunk + 1);
                count = 0;
                if(!last)
                    first = last = chunk;
                else
                    last = last->next = chunk;
            }
            last->attr[count*2] = attrname->str.ptr;
        }

        if(last)
        {
            CvFileNode stub;

            if(*ptr != '=')
                ptr = icvXMLSkipSpaces(fs, ptr, CV_XML_INSIDE_TAG);

            c = *++ptr;
            if( c != '\"' && c != '\'' )
                ptr = icvXMLSkipSpaces( fs, ptr, CV_XML_INSIDE_TAG );

            ptr = icvXMLParseValue(fs, ptr, &stub, CV_NODE_STRING);
            assert(stub.tag == CV_NODE_STRING);
            last->attr[count*2+1] = stub.data.str.ptr;
            count++;
        }

        c = *ptr;
        have_space = cv_isspace(c) || c == '\0';

        if(c != '>')
        {
            ptr = icvXMLSkipSpaces(fs, ptr, CV_XML_INSIDE_TAG);
            c = *ptr;
        }

        if(c == '>')
        {
            ptr++;
            break;
        }
        else if(c == '?' && tag_type == CV_XML_HEADER_TAG)
        {
            ptr += 2;
            break;
        }
        else if( c == '/' && ptr[1] == '>' && tag_type == CV_XML_OPENING_TAG )  // FIXIT ptr[1] - out of bounds read without check
        {
            tag_type = CV_XML_EMPTY_TAG;
            ptr += 2;
            break;
        }
    }
    *_tag = tagname;
    *_tag_type = tag_type;
    *_list = first;

    return ptr;
}

void icvXMLParse(CvFileStorage* fs)
{
    char* ptr = fs->buffer_start;
    StringHashNode *key = 0, *key2 = 0;
    AttrList* list = 0;
    int tag_type = 0;

    //prohibit leading comments
    ptr = icvXMLSkipSpaces(fs, ptr, CV_XML_INSIDE_TAG);

    ptr = icvXMLParseTag(fs, ptr, &key, &list, &tag_type);

    while(*ptr != '\0')
    {
        ptr = icvXMLSkipSpaces(fs, ptr, 0);

        if( *ptr != '\0' )
        {
            CvFileNode* root_node;
            ptr = icvXMLParseTag( fs, ptr, &key, &list, &tag_type );

            root_node = (CvFileNode*)SeqPush(fs->roots, 0);
            ptr = icvXMLParseValue(fs, ptr, root_node, CV_NODE_NONE);
            ptr = icvXMLParseTag(fs, ptr, &key2, &list, &tag_type);
            ptr = icvXMLSkipSpaces(fs, ptr, 0);
        }
    }
}

void fastFree(void* ptr)
{
    if(ptr)
    {
        uchar* udata = ((uchar**)ptr)[-1];
        free(udata);
    }
}

void cvFree_(void* ptr)
{
    fastFree(ptr);
}

void icvCloseFile(CvFileStorage* fs)
{
    if( fs->file )
        fclose( fs->file );

    fs->file = 0;
    fs->gzfile = 0;
    fs->strbuf = 0;
    fs->strbufpos = 0;
    fs->is_opened = false;
}

inline char* cv_skip_BOM(char* ptr)
{
    if((unsigned char)ptr[0] == 0xef && (unsigned char)ptr[1] == 0xbb && (unsigned char)ptr[2] == 0xbf) //UTF-8 BOM
    {
      return ptr + 3;
    }
    return ptr;
}

CvFileStorage* cvOpenFileStorage(const char* query, CvMemStorage* dststorage, int flags, const char* encoding)
{ // flags = 0
    CvFileStorage* fs = 0;
    int default_block_size = 1 << 18;
    bool append = (flags & 3) == CV_STORAGE_APPEND; //false
    bool mem = (flags & CV_STORAGE_MEMORY) != 0; //false
    bool write_mode = (flags & 3) != 0; //false
    bool isGZ = false;
    size_t fnamelen = 0;
    const char* filename = query;

    fnamelen = strlen(filename);

    fs = cvAlloc(sizeof(*fs));
    assert(fs);
    memset(fs, 0, sizeof(*fs));

    fs->memstorage = CreateMemStorage( default_block_size );
    fs->dststorage = dststorage ? dststorage : fs->memstorage;

    fs->flags = CV_FILE_STORAGE;
    fs->write_mode = write_mode;

    if(!mem)
    {
        fs->filename = (char*)MemStorageAlloc(fs->memstorage, fnamelen+1);
        strcpy(fs->filename, filename);

        char compression = '\0';

        fs->file = fopen(fs->filename, !fs->write_mode ? "rt" : !append ? "wt" : "a+t" );
    }

    fs->roots = 0;
    fs->struct_indent = 0;
    fs->struct_flags = 0;
    fs->wrap_margin = 71;

    fs->fmt = CV_STORAGE_FORMAT_XML;

    size_t buf_size = 1 << 20;
    char buf[16];
    icvGets(fs, buf, sizeof(buf)-2);
    char* bufPtr = cv_skip_BOM(buf);
    size_t bufOffset = bufPtr - buf;

    fseek(fs->file, 0, SEEK_END);
    buf_size = ftell(fs->file);

    buf_size = min(buf_size, (size_t)(1 << 20));
    buf_size = max(buf_size, (size_t)(CV_FS_MAX_LEN*2 + 1024));

    rewind(fs->file);
    fs->strbufpos = bufOffset;

    fs->str_hash = CreateMap(0, sizeof(StringHash), sizeof(StringHashNode), fs->memstorage, 256);
    fs->roots = CreateSeq(0, sizeof(Seq), sizeof(CvFileNode), fs->memstorage);

    fs->buffer = fs->buffer_start = (char*)cvAlloc(buf_size + 256);
    fs->buffer_end = fs->buffer_start + buf_size;
    fs->buffer[0] = '\n';
    fs->buffer[1] = '\0';

    icvXMLParse(fs);

    // release resources that we do not need anymore
    cvFree(&fs->buffer_start);
    fs->buffer = fs->buffer_end = 0;

    icvCloseFile(fs);
    // we close the file since it's not needed anymore. But icvCloseFile() resets is_opened,
    // which may be misleading. Since we restore the value of is_opened.
    fs->is_opened = true;
    return fs;
}

void reset(CvFileStorage* fs1, CvFileStorage* fs2)
{
    CvFileStorage temp = *fs1;
    *fs1 = *fs2;
    *fs2 = temp;
}

void isOpened(FileStorage fs)
{
    return fs.fs && fs.is_opened;
}

bool open(FileStorage* fs, char* filename, int flags)
{
    reset(fs->fs, cvOpenFileStorage(filename, 0, flags, 0));
    bool ok = isOpened(*fs);
    fs->state = ok ? NAME_EXPECTED + INSIDE_MAP : UNDEFINED;
    return ok;
}

FileStorage fileStorage(char* filename, int flags)
{
    FileStorage fs;
    fs.state = UNDEFINED;
    open(&fs, filename, flags);
    return fs;
}

CvFileNode* cvGetRootFileNode(const CvFileStorage* fs, int stream_index);
{
    if(!fs->roots || (unsigned)stream_index >= (unsigned)fs->roots->total)
        return 0;

    return (CvFileNode*)GetSeqElem(fs->roots, stream_index);
}

FileNode fileNode(const CvFileStorage* _fs, const CvFileNode* _node)
{
    FileNode fn;
    fn.fs = _fs;
    fn.node = _node;
    return fn;
}

FileNode root(FileStorage fs, int streamidx)
{
    return fileNode(fs.fs, cvGetRootFileNode(fs.fs, 0));
}

typedef struct FileNodeIterator
{
    const CvFileStorage* fs;
    const CvFileNode* container;
    SeqReader reader;
    size_t remaining;
} FileNodeIterator ;

SeqReader init_SeqReader()
{
    SeqReader sr;
    sr.header_size = 0;
    sr.seq = 0;
    sr.block = 0;
    sr.ptr = 0;
    sr.block_min = 0
    sr.block_max = 0;
    sr.delta_index = 0;
    sr.prev_elem = 0;
    return sr;
}

size_t size(FileNode fn)
{
    return (size_t)((Set*)fn.node->data.map)->active_count;
}

FileNodeIterator fileNodeIterator(const CvFileStorage* _fs, const CvFileNode* _node, size_t _ofs)
{
    FileNodeIterator fn_it;
    fn_it.reader = init_SeqReader();
    if(_fs && _node && CV_NODE_TYPE(_node->tag) != CV_NODE_NONE)
    {
        int node_type = _node->tag & 7;
        fn_it.fs = _fs;
        fn_it.container = _node;
        if(!(_node->tag & 16) && (node_type == 5 || node_type == 6))
        {
            StartReadSeq(_node->data.seq, (SeqReader*)&reader);
            remaining = size(fileNode(_fs, _node));
        }
    }
}

FileNodeIterator begin(FileNode r)
{
    return fileNodeIterator(r.fs, r.node, 0);
}

inline FileNode getFirstTopLevelNode(FileStorage fs) const 
{ 
    FileNode r = root(fs, 0);
    FileNodeIterator it = begin(r);
    return fileNode(it.fs, (const CvFileNode*)(const void*)it.reader.ptr);
}

typedef struct BoostTreeParams
{
    int boostType;
    int weakCount;
    double weightTrimRate;
} BoostTreeParams ;

typedef struct TrainData
{
} TrainData ;

typedef struct WorkData
{
    TrainData* data; //No fields, only funcs.
    vector* wnodes; //WNode
    vector* wsplits; //WSplit
    vector* wsubsets; //int
    vector* cv_Tn; //double
    vector* cv_node_risk; //double
    vector* cv_node_error; //double
    vector* cv_labels; //int
    vector* sample_weights; //double
    vector* cat_responses; //int
    vector* ord_responses; //double
    vector* sidx; //int
    int maxSubsetSize;
} WorkData ;

typedef struct Node
{
    double value; //!< Value at the node: a class label in case of classification or estimated
                      //!< function value in case of regression.
    int classIdx; //!< Class index normalized to 0..class_count-1 range and assigned to the
                  //!< node. It is used internally in classification trees and tree ensembles.
    int parent; //!< Index of the parent node
    int left; //!< Index of the left child node
    int right; //!< Index of right child node
    int defaultDir; //!< Default direction where to go (-1: left or +1: right). It helps in the
                    //!< case of missing values.
    int split; //!< Index of the first split
} Node ;

typedef struct Split
{
    int varIdx; //!< Index of variable on which the split is created.
    bool inversed; //!< If true, then the inverse split rule is used (i.e. left and right
                   //!< branches are exchanged in the rule expressions below).
    float quality; //!< The split quality, a positive number. It is used to choose the best split.
    int next; //!< Index of the next split in the list of splits for the node
    float c; /**< The threshold value in case of split on an ordered variable.
                  The rule is:
                  @code{.none}
                  if var_value < c
                    then next_node <- left
                    else next_node <- right
                  @endcode */
    int subsetOfs; /**< Offset of the bitset used by the split on a categorical variable.
                        The rule is:
                        @code{.none}
                        if bitset[var_value] == 1
                            then next_node <- left
                            else next_node <- right
                        @endcode */
} Split ;

typedef struct DTreesImplForBoost
{
    TreeParams params;

    vector* varIdx; //int
    vector* compVarIdx; //int
    vector* varType; //unsigned char
    vector* catOfs; //Point
    vector* catMap; //int
    vector* roots; //int
    vector* nodes; //Node
    vector* splits; //Split
    vector* subsets; //int
    vector* classLabels; //int
    vector* missingSubst; //float
    vector* varMapping; //int
    bool _isClassifier;

    WorkData* w;

    BoostTreeParams bparams;
    vector* sumResult; //double
} DTreesImplForBoost ;

typedef struct Boost
{
    DTreesImplForBoost impl;
} Boost;

bool haveCommonRegion_triplet(region_triplet t1, region_triplet t2)
{
    if( (t1.a.x ==t2.a.x && t1.a.y == t2.a.y) || (t1.a.x ==t2.b.x && t1.a.y == t2.b.y) || (t1.a.x ==t2.c.x && t1.a.y == t2.c.y) ||
        (t1.b.x ==t2.a.x && t1.b.y == t2.a.y) || (t1.b.x ==t2.b.x && t1.b.y == t2.b.y) || (t1.b.x ==t2.c.x && t1.b.y == t2.c.y) ||
        (t1.c.x ==t2.a.x && t1.c.y == t2.a.y) || (t1.c.x ==t2.b.x && t1.c.y == t2.b.y) || (t1.c.x ==t2.b.x && t1.c.y == t2.b.y) )
        return true;

    retur false;
}

// Check if two sequences share a region in common
bool haveCommonRegion(region_sequence sequence1, region_sequence sequence2)
{
    for (size_t i=0; i < vector_size(sequence2.triplets); i++)
    {
        for (size_t j=0; j < vector_size(sequence1.triplets); j++)
        {
            if (haveCommonRegion_triplet(*(region_triplet*)vector_get(sequence2.triplets, i), *(region_triplet*)vector_get(sequence1.triplets, j)));
                return true;
        }
    }

    return false;
}

CvFileNode* cvGetFileNodeByName(const CvFileStorage* fs, const CvFileNode* _map_node, const char* str)
{
    CvFileNode* value = 0;
    int i, len, tab_size;
    unsigned hashval = 0;
    int k = 0, attempts = 1;

    if(!fs)
        return 0;

    for(i = 0; str[i] != '\0'; i++)
        hashval = hashval*CV_HASHVAL_SCALE + (unsigned char)str[i];
    hashval &= INT_MAX;
    len = i;

    if(!_map_node)
    {
        if( !fs->roots )
            return 0;
        attempts = fs->roots->total;
    }

    for(k = 0; k < attempts; k++)
    {
        StringHash* map;
        const CvFileNode* map_node = _map_node;
        FileMapNode* another;

        if(!map_node)
            map_node = (CvFileNode*)GetSeqElem(fs->roots, k);

        if(!CV_NODE_IS_MAP(map_node->tag))
            return 0;

        map = map_node->data.map;
        tab_size = map->tab_size;

        if( (tab_size & (tab_size - 1)) == 0 )
            i = (int)(hashval & (tab_size - 1));
        else
            i = (int)(hashval % tab_size);

        for(another = (FileMapNode*)(map->table[i]); another != 0; another = another->next)
        {
            const StringHashNode* key = another->key;

            if(key->hashval == hashval &&
                key->str.len == len &&
                memcmp( key->str.ptr, str, len ) == 0)
            {
                value = &another->value;
                return value;
            }
        }
    }

    return value;
}

FileNode FileNode_op(const FileNode fn, const char* nodename) const
{
    return fileNode(fn.fs, cvGetFileNodeByName(fn.fs, fn.node, nodename));
}

void read(FileNode node, int* value, int default_value)
{
    *value = node.node->data.i;
}

int int_op(FileNode fn)
{
    int* value = malloc(sizeof(int));
    read(fn, value, 0);
    return *value;
}

typedef struct TreeParams
{
    bool  useSurrogates;
    bool  use1SERule;
    bool  truncatePrunedTree;
    Mat priors;

    int   maxCategories;
    int   maxDepth;
    int   minSampleCount;
    int   CVFolds;
    float regressionAccuracy;
} TreeParams ;

TreeParams init_TreeParams()
{
    TreeParams p;
    p.maxDepth = INT_MAX;
    p.minSampleCount = 10;
    p.regressionAccuracy = 0.01f;
    p.useSurrogates = false;
    p.maxCategories = 10;
    p.CVFolds = 10;
    p.use1SERule = true;
    p.truncatePrunedTree = true;
    Mat* m = &(p.priors);
    m->rows = m->cols = 0;
    m->flags = MAGIC_VAL;
    m->data = 0;
    m->datastart = 0;
    m->datalimit = 0;
    m->dataend = 0;
    m->step = 0;
    return p;
}

void ChangeSeqBlock(void* _reader, int direction)
{
    SeqReader* reader = (SeqReader*)_reader;

    if( direction > 0 )
    {
        reader->block = reader->block->next;
        reader->ptr = reader->block->data;
    }
    else
    {
        reader->block = reader->block->prev;
        reader->ptr = CV_GET_LAST_ELEM( reader->seq, reader->block );
    }
    reader->block_min = reader->block->data;
    reader->block_max = reader->block_min + reader->block->count * reader->seq->elem_size;
}

static int icvSymbolToType(char c)
{
    static const char symbols[9] = "ucwsifdr";
    const char* pos = strchr( symbols, c );
    return (int)(pos - symbols);
}

int icvDecodeFormat( const char* dt, int* fmt_pairs, int max_len )
{
    int fmt_pair_count = 0;
    int i = 0, k = 0, len = dt ? (int)strlen(dt) : 0;

    if( !dt || !len )
        return 0;

    assert( fmt_pairs != 0 && max_len > 0 );
    fmt_pairs[0] = 0;
    max_len *= 2;

    for( ; k < len; k++ )
    {
        char c = dt[k];

        if( cv_isdigit(c) )
        {
            int count = c - '0';
            if( cv_isdigit(dt[k+1]) )
            {
                char* endptr = 0;
                count = (int)strtol( dt+k, &endptr, 10 );
                k = (int)(endptr - dt) - 1;
            }

            fmt_pairs[i] = count;
        }
        else
        {
            int depth = icvSymbolToType(c);
            if( fmt_pairs[i] == 0 )
                fmt_pairs[i] = 1;
            fmt_pairs[i+1] = depth;
            if( i > 0 && fmt_pairs[i+1] == fmt_pairs[i-1] )
                fmt_pairs[i-2] += fmt_pairs[i];
            else
                i += 2;
            fmt_pairs[i] = 0;
        }
    }

    fmt_pair_count = i/2;
    return fmt_pair_count;
}

int icvCalcElemSize(const char* dt, int initial_size)
{
    int size = 0;
    int fmt_pairs[128], i, fmt_pair_count;
    int comp_size;

    fmt_pair_count = icvDecodeFormat( dt, fmt_pairs, 128);
    fmt_pair_count *= 2;
    for( i = 0, size = initial_size; i < fmt_pair_count; i += 2 )
    {
        comp_size = CV_ELEM_SIZE(fmt_pairs[i+1]);
        size = Align( size, comp_size );
        size += comp_size * fmt_pairs[i];
    }
    if( initial_size == 0 )
    {
        comp_size = CV_ELEM_SIZE(fmt_pairs[1]);
        size = Align( size, comp_size );
    }
    return size;
}

int icvCalcStructSize(const char* dt, int initial_size)
{
    int size = icvCalcElemSize(dt, initial_size);
    size_t elem_max_size = 0;
    for (const char * type = dt; *type != '\0'; type++) {
        switch (*type)
        {
        case 'u': { elem_max_size = max( elem_max_size, sizeof(uchar ) ); break; }
        case 'c': { elem_max_size = max( elem_max_size, sizeof(schar ) ); break; }
        case 'w': { elem_max_size = max( elem_max_size, sizeof(ushort) ); break; }
        case 's': { elem_max_size = max( elem_max_size, sizeof(short ) ); break; }
        case 'i': { elem_max_size = max( elem_max_size, sizeof(int   ) ); break; }
        case 'f': { elem_max_size = max( elem_max_size, sizeof(float ) ); break; }
        case 'd': { elem_max_size = max( elem_max_size, sizeof(double) ); break; }
        default: break;
        }
    }
    size = Align(size, (int)(elem_max_size));
    return size;
}

void cvReadRawDataSlice(const FileStorage* fs, SeqReader* reader, int len, void* _data, const char* dt)
{
    char* data0 = (char*)_data;
    int fmt_pairs[256], k = 0, fmt_pair_count;
    int i = 0, count = 0;

    fmt_pair_count = icvDecodeFormat(dt, fmt_pairs, 128);
    size_t step = icvCalcStructSize(dt, 0);

    for(;;)
    {
        int offset = 0;
        for(k = 0; k < fmt_pair_count; k++)
        {
            int elem_type = fmt_pairs[k*2+1];
            int elem_size = CV_ELEM_SIZE(elem_type);
            char* data;

            count = fmt_pairs[k*2];
            offset = Align(offset, elem_size);
            data = data0 + offset;

            for(i = 0; i < count; i++)
            {
                CvFileNode* node = (CvFileNode*)reader->ptr;
                if( CV_NODE_IS_INT(node->tag) )
                {
                    int ival = node->data.i;
                    switch( elem_type )
                    {
                    case 0:
                        *(unsigned char*)data = (unsigned char)(ival);
                        data++;
                        break;
                    case 1:
                        *(char*)data = (signed char)(ival);
                        data++;
                        break;
                    case 2:
                        *(ushort*)data = (unsigned short)(ival);
                        data += sizeof(ushort);
                        break;
                    case 3:
                        *(short*)data = (short)(ival);
                        data += sizeof(short);
                        break;
                    case 4:
                        *(int*)data = ival;
                        data += sizeof(int);
                        break;
                    case 5:
                        *(float*)data = (float)ival;
                        data += sizeof(float);
                        break;
                    case 6:
                        *(double*)data = (double)ival;
                        data += sizeof(double);
                        break;
                    case 7: /* reference */
                        *(size_t*)data = ival;
                        data += sizeof(size_t);
                        break;
                    }
                }

                 else if(CV_NODE_IS_REAL(node->tag))
                {
                    double fval = node->data.f;
                    int ival;

                    switch(elem_type)
                    {
                    case 0:
                        ival = round(fval);
                        *(uchar*)data = (unsigned char)(ival);
                        data++;
                        break;
                    case 1:
                        ival = round(fval);
                        *(char*)data = (signed char)(ival);
                        data++;
                        break;
                    case 2:
                        ival = round(fval);
                        *(ushort*)data = (unsigned short)(ival);
                        data += sizeof(ushort);
                        break;
                    case 3:
                        ival = round(fval);
                        *(short*)data = (short)(ival);
                        data += sizeof(short);
                        break;
                    case 4:
                        ival = round(fval);
                        *(int*)data = ival;
                        data += sizeof(int);
                        break;
                    case 5:
                        *(float*)data = (float)fval;
                        data += sizeof(float);
                        break;
                    case 6:
                        *(double*)data = fval;
                        data += sizeof(double);
                        break;
                    case 7: /* reference */
                        ival = round(fval);
                        *(size_t*)data = ival;
                        data += sizeof(size_t);
                        break;
                    }
                }

                CV_NEXT_SEQ_ELEM(sizeof(CvFileNode), *reader);
                if(!--len)
                    goto end_loop;
            }

            offset = (int)(data - data0);
        }
        data0 += step;
    }

end_loop:
    if(!reader->seq)
        reader->ptr -= sizeof(CvFileNode);
}

static void getElemSize(const char* fmt, size_t* elemSize, size_t* cn)
{
    const char* dt = fmt;
    elemSize = malloc(sizeof(size_t));
    cn = malloc(sizeof(size_t));

    *cn = 1;
    if(cv_isdigit(dt[0]))
    {
        *cn = dt[0] - '0';
        dt++;
    }
    char c = dt[0];
    *elemSize = (*cn)*(c == 'u' || c == 'c' ? sizeof(unsigned char) : c == 'w' || c == 's' ? sizeof(unsigned short) :
        c == 'i' ? sizeof(int) : c == 'f' ? sizeof(float) : c == 'd' ? sizeof(double) :
        c == 'r' ? sizeof(void*) : (size_t)0);
}

void readRaw(FileNodeIterator* it, const char* fmt, unsigned char* vec, size_t maxCount)
{
    if(it->fs && it->container && it->remaining > 0)
    {
        size_t elem_size, cn;
        getElemSize(fmt, &elem_size, &cn);
        size_t count = min(it->remaining, maxCount);

        cvReadRawDataSlice(it->fs, (SeqReader*)(&(it->reader)), (int)count, vec, fmt);
        it->remaining -= count*cn;
    }
}

void VecReader_op(FileNodeIterator* it, vector* v/*int*/, size_t count, code)
{
    size_t remaining = it->remaining;
    size_t cn = 1;
    int _fmt = 0;
    if(code == VECTOR_INT)
        _fmt = 105;

    if(code == VECTOR_UCHAR)
        _fmt = 117;

    char fmt[] = { (char)((_fmt >> 8)+'1'), (char)_fmt, '\0' };
    size_t remaining1 = remaining / cn;
    count = count < remaining1 ? count : remaining1;
    vector_resize(v, count);
    if(code == VECTOR_INT)
        readRaw(it, fmt, (unsigned char*)vector_get(vec, 0), count*sizeof(int));

    if(code == VECTOR_UCHAR)
        readRaw(it, fmt, (unsigned char*)vector_get(v, 0), count*sizeof(unsigned char));
}

void rightshift_op_(FileNodeIterator* it, vector* v/*int*/, int code)
{
    
    FileNodeIterator it_;
    VecReader_op(it, v, INT_MAX, code);
}



void rightshift_op(const FileNode node, vector* v/*int*/, int code)
{
    FileNodeIterator it = begin(node);
    rightshift_op_(&it, v, code);
}

static inline void readVectorOrMat(const FileNode node, vector* v/*int*/, int code)
{
    rightshift_op(node, v, code);
}

void initCompVarIdx(DTreesImplForBoost* impl)
{
    int nallvars = vector_size(impl->varType);
    vector_resize(impl->compVarIdx, nallvars);
    int push_val = -1;
    for(int i = 0; i < nallvars; i++)
        vector_add(impl->compVarIdx, &push_val);
    int i, nvars = vector_size(impl->varIdx), prevIdx = -1;
    for(i = 0; i < nvars; i++)
    {
        int vi = *(int*)vector_get(impl->varIdx, i);
        assert(0 <= vi && vi < nallvars && vi > prevIdx);
        prevIdx = vi;
        vector_set(impl->compVarIdx, vi, &i);
    }
}

void readParams_(DTreesImplForBoost* impl, const FileNode fn)
{
    impl->_isClassifier = false;
    FileNode tparams_node = FileNode_op(fn, "training_params");

    TreeParams param0 = init_TreeParams();
    param0.useSurrogates = false;
    param0.maxCategories = 10;
    param0.regressionAccuracy = 0.00999999978;
    param0.maxDepth = 1;
    param0.minSampleCount = 10;
    param0.CVFolds = 0;
    param0.use1SERule = true;
    param0.truncatePrunedTree = true;
    param0.use1SERule = true;   
    release(&(param0.priors));

    readVectorOrMat(FileNode_op(fn, "var_idx"), impl->varIdx, VECTOR_INT);
    rightshift_op(FileNode_op(fn, "varType"), impl->varType, VECTOR_UCHAR);   

    bool isLegacy= true;

    int varAll = int_op(FileNode_op(fn, "var_all"));

    if(isLegacy && vector_size(impl->varType) <= varAll)
    {
        vector* extendedTypes = malloc(sizeof(vector)); //unsigned char
        vector_init_n(extendedTypes, varAll+1);
        unsigned char push_val = (unsigned char)0;
        for(int i = 0; i < varAll+1; i++)
            vector_add(extendedTypes, &push_val);

        int i = 0, n;
        if(!vector_empty(impl->varIdx))
        {
            n = vector_size(impl->varIdx);
            for (; i < n; ++i)
            {
                int var = *(int*)vector_get(varIdx, i);
                vector_set(extendedTypes, var, vector_get(impl->varType, i));
            }
        }
        push_val = (unsigned char)1;
        vector_set(extendedTypes, varAll, &push_val);
        vector_swap(extendedTypes, impl->varType);
    }

    readVectorOrMat(FileNode_op(fn, "cat_map"), impl->catMap, VECTOR_INT);

    if(isLegacy)
    {
        // generating "catOfs" from "cat_count"
        vector_init(impl->catOfs);
        vector_init(impl->classLabels);
        vector* counts; //int
        vector_init(counts);
        readVectorOrMat(fn["cat_count"], counts, VECTOR_INT);
        unsigned int i = 0, j = 0, curShift = 0, size = (int)vector_size(impl->varType) - 1;
        for(; i < size; ++i)
        {
            Point newOffsets = init_Point(0, 0);
            if((*(int*)vector_get(impl->varType, i)) == 1) // only categorical vars are represented in catMap
            {
                newOffsets.x = curShift;
                curShift += (*(int*)vector_get(counts, j));
                newOffsets.y = curShift;
                ++j;
            }
            vector_add(impl->catOfs, &newOffsets);
        }
    }

    vector_assign(impl->varIdx, impl->varMapping);
    initCompVarIdx(impl);
    impl->params = params0;
}

void getNextIterator_(FileNodeIterator* it)
{
    if(it->remaining > 0)
    {
        if(it->reader.seq)
        {
            if(((it->reader).ptr += (((Seq*)it->reader.seq)->elem_size)) >= (it->reader).block_max)
            {
                ChangeSeqBlock((SeqReader*)&(it->reader), 1);
            }
        }
        it->remaining--;
    }
}

void readParams(DTreesImplForBoost* impl, const FileNode fn)
{
    readParams_(impl, fn);
    FileNode tparams_node = FileNode_op(fn, "training_params");
    // check for old layout
    impl->bparams.boostType  = 1;
    impl->bparams.weightTrimRate = 0.0;
}

void read_double(FileNode node, double* value, double default_value)
{
    *value = !node.node ? default_value :
        CV_NODE_IS_INT(node.node->tag) ? (float)node.node->data.i : (float)(node.node->data.f);
}

double double_op(const FileNode node)
{
    *value = !node.node ? default_value :
        CV_NODE_IS_INT(node.node->tag) ? (double)node.node->data.i : (double)(node.node->data.f);
}

void read_float(FileNode node, float* value, float default_value)
{
    *value = !node.node ? default_value :
        CV_NODE_IS_INT(node.node->tag) ? (float)node.node->data.i : (float)(node.node->data.f);
}

float float_op(const FileNode fn)
{
    float* value = malloc(sizeof(float));
    read_float(fn, value, 0.f);
    return *value;
}

int readSplit(DTreesImplForBoost* impl, const FileNode fn)
{
    Split split;

    int vi = int_op(fileNode(fn, "var"));
    vi = *(int*)vector_get(impl->varMapping, vi); // convert to varIdx if needed
    split.varIdx = vi;

    FileNode cmpNode = FileNode_op(fn, "le");
    if(!cmpNode.node)
    {
        cmpNode = FileNode_op(fn, "gt");
        split.inversed = true;
    }
    split.c = float_op(cmpNode);

    split.quality = float_op(fn, "quality");
    vector_add(impl->splits, &split);

    return vector_size(impl->splits)-1;
}

int readNode(DTreesImplForBoost* impl, const FileNode fn)
{
    Node node;
    node.value = double_op(FileNode_op(fn, "value"));

    if(impl->_isClassifier)
        node.classIdx = int_op(FileNode_op(fn, "norm_class_idx"));

    FileNode sfn = FileNode_op(fn, "splits");
    if(sfn.node != 0)
    {
        int i, n = sfn.node->data.seq->total, prevsplit = -1;
        FileNodeIterator it = begin(sfn);

        for(i = 0; i < n; i++, getNextIterator_(&it))
        {
            int splitidx = readSplit(impl, fileNode(it.fs, (const CvFileNode*)(const void*)it.reader.ptr));
            if(splitidx < 0)
                break;
            if(prevsplit < 0)
                node.split = splitidx;
            else
                ((Split*)vector_get(impl->splits, prevsplit))->next = splitidx;
            prevsplit = splitidx;
        }
    }
}

int readTree(DTreesImplForBoost* impl, const FileNode fn)
{
    int i, n = (size_t)fn.node->data.seq->total, root = -1, pidx = -1;
    FileNodeIterator it = begin(fn);

    for(i = 0; i < n; i++, getNextIterator_(&t))
    {
        int nidx = readNode(impl, fileNode(it.fs, (const CvFileNode*)(const void*)it.reader.ptr));
        if(nidx < 0)
            break;
        Node* node = vector_get(impl->nodes, nidx);
        node->parent = pidx;
        if(pidx < 0)
            root = nidx;
        else
        {
            Node* parent = vector_get(impl->nodes, pidx);
            if(parent->left < 0)
                parent->left = nidx;
            else
                parent->right = nidx;
        }
        if(node->split >= 0)
            pidx = nidx;
        else
        {
            while(pidx >= 0 && ((Node*)vector_get(impl->nodes, pidx))->right >= 0)
                pidx = ((Node*)vector_get(impl->nodes, pidx))->parent;
        }
    }
    vector_add(impl->roots, &root);
    return root;
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

AutoBuffer init_AutoBuffer(size_t _size)
{
    AutoBuffer ab;
    ab.fixed_size = 1024/sizeof(int)+8;
    ab.buf = (int*)malloc((ab.fixed_size > 0) ? ab.fixed_size : 1);
    ab.ptr = ab.buf;
    ab.sz = ab.fixed_size;
    allocateAB(&ab, _size);
    return ab;
}

AutoBuffer init_AutoBufferPointer(size_t _size)
{
    AutoBuffer ab;
    ab.fixed_size = 1024/sizeof(Point*)+8;
    ab.buf = (Point**)malloc((ab.fixed_size > 0) ? ab.fixed_size : 1);
    ab.ptr = ab.buf;
    ab.sz = ab.fixed_size;
    allocateAB(&ab, _size);
    return ab;
}

AutoBuffer init_AutoBufferPoint(size_t _size)
{
    AutoBuffer ab;
    ab.fixed_size = 1024/sizeof(int)+8;
    ab.buf = (Point*)malloc((ab.fixed_size > 0) ? ab.fixed_size : 1);
    ab.ptr = ab.buf;
    ab.sz = ab.fixed_size;
    allocateAB(&ab, _size);
    return ab;
}

AutoBuffer init_AutoBufferdouble(size_t _size)
{
    AutoBuffer ab;
    ab.fixed_size = 1024/sizeof(double)+8;
    ab.buf = (double*)malloc((ab.fixed_size > 0) ? ab.fixed_size : 1);
    ab.ptr = ab.buf;
    ab.sz = ab.fixed_size;
    allocateAB(&ab, _size);
    return ab;
}

AutoBuffer init_AutoBufferPt(size_t _size)
{
    AutoBuffer ab;
    ab.fixed_size = 1024/sizeof(int)+8;
    ab.buf = (PtInfo*)malloc((ab.fixed_size > 0) ? ab.fixed_size : 1);
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
    if(_size > ab->fixed_size)
        ab->ptr = malloc(_size);
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

inline void AutoBuffer_resize(AutoBuffer* ab, size_t _size)
{
    if(_size <= ab->sz)
    {
        ab->sz = _size;
        return;
    }
    size_t i, prevsize = sz, minsize = min(prevsize, _size);
    void* prevptr = ab->ptr;

    ab->ptr = _size > ab->fixed_size ? malloc(_size) : buf;
    ab->sz = _size;

    if(ptr != prevptr)
        for(i = 0; i < minsize; i++ )
            ab->ptr[i] = prevptr[i];
    for(i = prevsize; i < _size; i++)
        ab->ptr[i] = init_Point(0, 0);

    if(prevptr != ab->buf)
        free(prevptr);
}

typedef void (*BinaryFunc)(const unsigned char* src1, size_t step1,
                       const unsigned char* src2, size_t step2,
                       unsigned char* dst, size_t step, Point sz);

typedef void (*SplitFunc)(const unsigned char* src, unsigned char** dst, int len, int cn);

typedef int (*CountNonZeroFunc)(const unsigned char*, int);

int updateContinuityFlag(Mat* m)
{
    int i, j;
    int sz[] = {m->rows, m->cols};
    
    for(i = 0; i < 2; i++)
    {
        if(sz[i] > 1)
            break;
    }
    
    uint64_t t = (uint64_t)sz[i]*CV_MAT_CN(m->flags);
    for(j = 1; j > i; j--)
    {
        t *= sz[j];
        if(m->step[j]*sz[j] < m->step[j-1] )
            break;
    }

    if(j <= i && t == (uint64_t)(int)t)
        return m->flags | CONTINUOUS_FLAG;

    return m->flags & ~CONTINUOUS_FLAG;
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
    if(!m)
        return;
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
    int dtype = 0;
    
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

        Point sz = getContinuousSize(src, dst, 1);
        size_t len = sz.x*elemSize(src);

        for(; sz.y-- ; sptr += src.step, dptr += dst->step)
            memcpy(dptr, sptr, len);
    }
    return;
}

Point tl(Rect rect)
{
    return init_Point(rect.x, rect.y);
}

Point br(Rect rect)
{
    return init_Point(rect.x + rect.width, rect.y + rect.height);
}

static void cpy_8u(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep, Point size)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for(; size.y--; src += sstep, dst += dstep)
        memcpy(dst, src, size.x*sizeof(src[0]));
}

void cvtScale_8u(const unsigned char* src, size_t sstep, const unsigned char* dst, size_t dstep, Point size, float scale, float shift)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for(;size.y--; src += sstep, dst += dstep)
    {
        int x = 0;

        for(; x < size.x; x++ )
            dst[x] = (unsigned char)(src[x]*scale + shift);
    }
}

void cvt32f8u(const float* src, size_t sstep, unsigned char* dst, size_t dstep, Point size)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for(; size.height--; src += sstep, dst += dstep)
    {
        int x = 0;
        for(; x < size.width; x++)
            dst[x] = (unsigned char)src[x];
    }
}

void convertTo(Mat src, Mat* dst, int _type, double alpha/*1*/, double beta/*0*/, int code)
{
    if(empty(src))
    {
        release(dst);
        return;
    }

    bool noScale = fabs(alpha-1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;
    if(_type < 0)
        _type = type(src);
    else
         _type = CV_MAKETYPE(CV_MAT_DEPTH(_type), channels(src));

    int sdepth = depth(src), ddepth = CV_MAT_DEPTH(_type);
    if(sdepth == ddepth && noScale)
    {
        copyTo(src, dst);
        return;
    }

    create(dst, src.rows, src.cols, _type);
    //BinaryFunc func = getConvertFunc(sdepth, ddepth);

    double scale[] = {alpha, beta};
    int cn = channels(src);

    Point sz = getContinuousSize(src, dst, cn);
    if(code == CVT32F8U)
    {
        cvt32f8u(src.data, src.step, dst->data, dst->step, sz);
        return;
    }

    if(code == CVTSCALE8U)
    {
        cvtScale_8u(src.data, src.step, dst->data, dst->step, sz, (float)scale[0], (float)scale[1]);
        return;
    }

    else
        cpy_8u(src.data, src.step, dst->data, dst->step, sz);
}

static inline int Align(int size, int align)
{
    return (size+align-1) & -align;
}

static inline unsigned char* alignptr(unsigned char* ptr, int n)
{
    return (unsigned char*)(((size_t)ptr + n-1) & -n);
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
    for(i = 0; i < cn; i++)
        buf[i] = s.val[i];

    for(; i < unroll_to; i++)
        buf[i] = buf[i-cn];
}

void scalartoRawData(const Scalar s, double* buf, int type, int unroll_to)
{
    int i;
    const int depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    for(i = 0; i < cn; i++)
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
    _type = CV_MAT_TYPE(_type);
    
    release(m);
    m->rows = _rows;
    m->cols = _cols;
    m->flags = (_type & CV_MAT_TYPE_MASK) | MAGIC_VAL;
    m->data = 0;

    size_t esz = CV_ELEM_SIZE(m->flags), total = esz;
    for(int i = 1; i >= 0; i--)
    {
        m->step[i] = total;
        signed __int64 total1 = (signed __int64)total*sz[i];
        if((unsigned __int64)total1 != (size_t)total1)
            fatal("The total matrix size does not fit to \"size_t\" type");
        total = (size_t)total1;
    }
    if(m->rows * m->cols > 0)
    {
        size_t total_ = CV_ELEM_SIZE(_type);
        for(int i = 1; i >= 0; i++)
        {
            if(m->step)
                m->step[i] = total_;

            total_ *= sizes[i];
        }
        unsigned char* udata = malloc(size + sizeof(void*) + CV_MALLOC_ALIGN);
        if(!udata)
            fatal("Out of Memory Error");
        unsigned char** adata = alignPtr((unsigned char**)udata + 1, CV_MALLOC_ALIGN);
        adata[-1] = udata;
        m->datastart = m->data = (unsigned char*)adata;
        assert(m->step[1] == (size_t)CV_ELEM_SIZE(flags));
    }
    m->flags = updateContinuityFlag(m);
    if(m->data)
    {
        m->datalimit = m->datastart + m->rows*m->step[0];
        if(m->rows > 0)
        {
            m->dataend = ptr(*m, 0) + m->cols*m->step[1];
            m->dataend += (m->rows - 1)*m->step[0];
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
            scalartoRawData(s, scalar, type(*m), 12);
            size_t blockSize = 12*elemSize1(*m);

            for(size_t j = 0; j < elsize; j += blockSize)
            {
                size_t sz = min(blockSize, elsize - j);
                memcpy(dptr+j, scalar, sz);
            }
        }

        for(size_t i = 1; i < it.nplanes; i++, getNextIterator(&it))
            memcpy(dptr, m->data, elsize);
    }
}

inline Mat createusingPoint(vector* v, bool copyData /* Default false */)
{
    Mat m;
    m.flags = MAGIC_VAL | 7 | CV_MAT_CONT_FLAG;
    m.rows = vector_size(v);
    m.cols = 1;
    m.data = m.datastart = m.dataend = m.datalimit = 0;
    if(vector_empty(v))
        return m;
    if(!copyData)
    {
        m.step[0] = m.step[1] = m.sizeof(Point);
        m.datastart = m.data = (unsigned char*)vector_get(v, 0);
        m.datalimit = m.dataend = m.datastart + m.rows * m.step[0];
    }

    else
        m = createMat(vector_size(v), 1, 7, (unsigned char*)vector_get(v, 0), AUTO_STEP);

    return m;
}

Mat row_op(Mat m, Point rowRange)
{
    Mat sample;
    sample.flags = m.flags;
    sample.rows = m.rows;
    sample.cols = m.cols;
    sample.step[0] = m.step[0];
    sample.step[1] = m.step[1];
    sample.data = m.data;
    sample.datastart = m.datastart;
    sample.dataend = m.dataend;
    sample.datalimit = m.datalimit;

    updateContinuityFlag(&sample);
}

void meanStdDev(Mat* src, Scalar* mean, Scalar* sdv, Mat* mask)
{
    int k, cn = channels(*src), depth = depth(*src);

    SumSqrFunc func = getSumSqrTab(depth);

    assert(func != 0);

    const Mat* arrays[] = {src, mask, 0};
    unsigned char* ptrs[2];
    MatIterator it = matIterator(arrays, 0, ptrs, 2);
    int total = it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0, nz0 = 0;
    AutoBuffer _buf = init_AutoBufferdouble(cn*4);
    double* s = malloc(sizeof(double)* cn*4), *sq = s + cn;
    int *sbuf = (int*)s, *sqbuf = (int*)sq;
    bool blockSum = depth <= CV_16S, blockSqSum = depth <= CV_8S;
    size_t esz = 0;

    for(k = 0; k < cn; k++)
        s[k] = sq[k] = 0;

    if(blockSum)
    {
        intSumBlockSize = 1 << 15;
        blockSize = min(blockSize, intSumBlockSize);
        sbuf = (int*)(sq + cn);
        if(blockSqSum)
            sqbuf = sbuf + cn;
        for(k = 0; k < cn; k++)
            sbuf[k] = sqbuf[k] = 0;
        esz = elemSize(*src);
    }

    for(size_t i = 0; i < it.nplanes; i++, getNextIterator(&it))
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = min(total - j, blockSize);
            int nz = func(ptrs[0], ptrs[1], (unsigned char*)sbuf, (unsigned char*)sqbuf, bsz, cn);
            count += nz;
            nz0 += nz;
            if(blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)))
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += sbuf[k];
                    sbuf[k] = 0;
                }
                if( blockSqSum )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        sq[k] += sqbuf[k];
                        sqbuf[k] = 0;
                    }
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
            if( ptrs[1] )
                ptrs[1] += bsz;
        }
    }

    double scale = nz0 ? 1./nz0 : 0.;
    for(k = 0; k < cn; k++)
    {
        s[k] *= scale;
        sq[k] = sqrt(max(sq[k]*scale - s[k]*s[k], 0.));
    }

    for(j = 0; j < 2; j++)
    {
        const double* sptr = j == 0 ? s : sq;
        Scalar* _dst = j == 0 ? mean : sdv;
        
        Mat dst = getMatfromScalar(_dst);

        int dcn = total(dst);
        assert(type(dst) == CV_64F && isContinuous(dst) &&
                   (dst.cols == 1 || dst.rows == 1) && dcn >= cn );
        double* dptr = (double*)ptr(dst, 0);
        for(k = 0; k < cn; k++)
            dptr[k] = sptr[k];
        for(; k < dcn; k++)
            dptr[k] = 0;
    }
}


Mat getMatfromScalar(Scalar* s)
{
    int flags = 3238133759;
    return createMat(4, 1, flags, s, AUTO_STEP);
}

void divide_op(Mat* m, double s)
{
    convertTo(*m, m, -1, 1.0/s, 0, CVTSCALE8U);
}

void minus_op(Mat* m, double s)
{

}

void zeros(Mat* m, int _rows, int _cols, int _type)
{
    *m = createMat(_rows, _cols, _type, (void*)(size_t)0xEEEEEEEE, AUTO_STEP);
    createusingScalar(m, init_Scalar(0, 0, 0, 0));     
}

Mat createMat(int _rows, int _cols, int _type, void* _data, size_t _step/* AUTO_STEP */) // _data = (void*)(size_t)0xEEEEEEEE for Mat::zeros
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

void createVectorOfVector(vector** v, int rows, int cols, int mtype, int i, bool allowTransposed, int fixedDepthMask)
{
    int sizebuf[2];
    mtype = CV_MAT_TYPE(mtype);

    size_t len = rows*cols > 0 ? rows + cols - 1 : 0;

    if(i < 0)
    {
        vector_resize(v, len);
        return;
    }


    int type0 = CV_MAT_TYPE(flags);
    int esz = CV_ELEM_SIZE(type0);

}

int countNonZero(Mat src)
{
    int type = type(src), cn = CV_MAT_CN(type);
    assert(cn == 1);

    CountNonZeroFunc func = getCountNonZeroTab(depth(src));
    assert(func != 0);

    const Mat* arrays[] = {&src, 0};
    unsigned char* ptrs[1];
    MatIterator it = matIterator(arrays, 0, ptrs, 1);
    int total = (int)it.size, nz = 0;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        nz += func(ptrs[0], total);

    return nz;
}

void createVectorMat(vector* v/* Mat */, int rows, int cols, int mtype, int i/* -1 */)
{
    mtype = CV_MAT_TYPE(mtype);

    if(i < 0)
    {
        assert(rows == 1 || cols == 1 || rows*cols == 0);
        size_t len = rows*cols > 0 ? rows + cols - 1 : 0;

        vector_resize(v, len);
        return;
    }

    assert(i < vector_size(v));
    Mat* m = (Mat*)vector_get(v, i);

    create(m, rows, cols, mtype);
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

CvtColorLoop_Invoker cvtColorLoop_Invoker(const unsigned char* src_data_, size_t src_step_, unsigned char* dst_data_, size_t dst_step_, int width_, const void* _cvt, int code_cvt)
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

void split_(const unsigned char* src, unsigned char** dst, int len, int cn)
{
    int k = cn % 4 ? cn % 4 : 4;
    int i, j;
    if(k == 1)
    {
        unsigned char* dst0 = dst[0];

        if(cn == 1)
        {
            memcpy(dst0, src, len*sizeof(unsigned char));
        }
        else
        {
            for(i = 0, j = 0 ; i < len; i++, j += cn)
                dst0[i] = src[j];
        }
    }
    else if(k == 2)
    {
        unsigned char *dst0 = dst[0], *dst1 = dst[1];
        i = j = 0;

        for(; i < len; i++, j += cn)
        {
            dst0[i] = src[j];
            dst1[i] = src[j+1];
        }
    }
    else if(k == 3)
    {
        unsigned char *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2];
        i = j = 0;

        for(; i < len; i++, j += cn)
        {
            dst0[i] = src[j];
            dst1[i] = src[j+1];
            dst2[i] = src[j+2];
        }
    }
    else
    {
        unsigned char *dst0 = dst[0], *dst1 = dst[1], *dst2 = dst[2], *dst3 = dst[3];
        i = j = 0;

        for(; i < len; i++, j += cn)
        {
            dst0[i] = src[j]; dst1[i] = src[j+1];
            dst2[i] = src[j+2]; dst3[i] = src[j+3];
        }
    }

    for(; k < cn; k += 4)
    {
        unsigned char *dst0 = dst[k], *dst1 = dst[k+1], *dst2 = dst[k+2], *dst3 = dst[k+3];
        for(i = 0, j = k; i < len; i++, j += cn)
        {
            dst0[i] = src[j]; dst1[i] = src[j+1];
            dst2[i] = src[j+2]; dst3[i] = src[j+3];
        }
    }
}

void split_(Mat src, Mat* mv)
{
    int k, depth = depth(src), cn = channels(src);
    if(cn == 1)
    {
        copyTo(src, &mv[0]);
        return;
    }

    for(k = 0;k < cn;k++)
        create(&mv[k], src.rows, src.cols, depth);

    size_t esz = elemSize(src), esz1 = elemSize1(src);
    size_t blocksize0 = (BLOCK_SIZE + esz-1)/esz;
    unsigned char* _buf = malloc((cn+1)*(sizeof(Mat*) + sizeof(unsigned char*)) + 16);
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

    for(size_t i = 0; i < it.nplanes; i++, getNextIterator(&it))
    {
        for(size_t j = 0; j < total; j += blocksize)
        {
            size_t bsz = min(total - j, blocksize);
            split_(it.ptrs[0], &it.ptrs[1], (int)bsz, cn);

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

    createVectorMat(dst, cn, 1, depth, -1);
    for (int i = 0; i < cn; ++i)
        createVectorMat(dst, m.rows, m.cols, depth, i);

    split_(m, vector_get(dst, 0));
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

Size size(int _width, int _height)
{
    Size sz;
    sz.width = _width;
    sz.height = _height;
    return sz;
}

floatPoint init_floatPoint(float _x, float _y)
{
    floatPoint fp;
    fp.x = _x;
    fp.y = _y;
    return fp;
}

CvtHelper cvtHelper(Mat _src, Mat* _dst, int dcn)
{
    CvtHelper h;
    int stype = type(src);
    h.scn = CV_MAT_CN(stype), h.depth = CV_MAT_DEPTH(stype);

    if(_src == _dst)
        copyTo(_src, &h.src);

    else
        h.src = _src;

    Size sz = size(h.src.cols, h.src.rows);
    h.dstSz = sz;
    create(&h.dst, h.src.rows, h.src.cols, CV_MAKETYPE(depth, dcn));
    create(_dst, h.src.rows, h.src.cols, CV_MAKETYPE(depth, dcn));
    return h;
}

void normalizeAnchor(Point* anchor, Point ksize)
{
    if(anchor->x < 0)
        anchor->x = ksize.y >> 1;
    
    if(anchor->y < 0)
        anchor->y = ksize.x >> 1;
}

typedef struct BaseFilter
{
    Point ksize;
    Point anchor;
    vector* coords; // Point
    vector* coeffs; //unsigned char
    std::vector<uchar*> ptrs; //unsigned char*
    float delta;
    FilterVec_32f vecOp;
} BaseFilter ;

typedef struct BaseDimFilter
{
    int ksize;
    int anchor;
} BaseDimFilter ;

typedef struct FilterEngine
{
    int srcType;
    int dstType;
    int bufType;
    Point ksize;
    Point anchor;
    int maxWidth;
    Point wholeSize;
    Rect roi;
    int dx1;
    int dx2;
    int rowBorderType;
    int columnBorderType;
    vector* borderTab; //int
    int borderElemSize;
    vector* ringBuf; //unsigned char
    vector* srcRow; //unsigned char
    vector* constBorderValue; //unsigned char
    vector* constBorderRow; //unsigned char
    int bufStep;
    int startY;
    int startY0;
    int endY;
    int rowCount;
    int dstY;
    vector* rows;// unsigned char*

    BaseFilter* filter2D;
    BaseDimFilter* rowFilter;
    BaseDimFilter* columnFilter;
} FilterEngine ;

void init_FilterEngine(FilterEngine* filter, BaseFilter* _filter2D, BaseDimFilter* _rowFilter, BaseDimFilter* _columnFilter,
                       int _srcType, int _dstType, int _bufType,
                       int _rowBorderType/* BORDER_REPLICATE */, int _columnBorderType/*-1*/, const Scalar _borderValue /* init_Scalar(0, 0, 0, 0) */)
{
    filter = malloc(sizeof(FilterEngine));
    filter->srcType = filter->dstType = filter->bufType = -1;
    filter->maxWidth = filter->dx1 = filter->dx2 = 0;
    filter->wholeSize = init_Point(-1, -1);
    filter->rowBorderType = filter->columnBorderType = BORDER_REPLICATE;
    filter->rows = malloc(sizeof(vector));
    vector_init(filter->rows);
    filter->borderElemSize = filter->bufStep = filter->startY = filter->startY0 = filter->rowCount = filter->dstY = 0;

    _srcType = CV_MAT_TYPE(_srcType);
    _bufType = CV_MAT_TYPE(_bufType);
    _dstType = CV_MAT_TYPE(_dstType);

    filter->srcType = _srcType;
    int srcElemSize = (int)CV_ELEM_SIZE(filter->srcType);
    filter->dstType = _dstType;
    filter->bufType = _bufType;

    filter->filter2D = _filter2D;
    filter->rowFilter = _rowFilter;
    filter->columnFilter = _columnFilter;

    if(_columnBorderType < 0)
        _columnBorderType = _rowBorderType;

    filter->rowBorderType = _rowBorderType;
    filter->columnBorderType = _columnBorderType;

    assert(columnBorderType != BORDER_WRAP);

    if(!filter->filter2D)
    {
        assert(filter->rowFilter && filter->columnFilter);
        filter->ksize = init_Point(rowFilter->ksize, columnFilter->ksize);
        filter->anchor = init_Point(rowFilter->anchor, columnFilter->anchor);
    }
    else
    {
        assert(filter->bufType == filter->srcType);
        filter->ksize = filter2D->ksize;
        filter->anchor = filter2D->anchor;
    }

    assert(0 <= anchor.x && anchor.x < ksize.x &&
           0 <= anchor.y && anchor.y < ksize.y);

    filter->borderElemSize = srcElemSize/(CV_MAT_DEPTH(filter->srcType) >= CV_32S ? sizeof(int) : 1);
    int borderLength = max(filter->ksize.x-1, 1);
    filter->borderTab = malloc(sizeof(vector));
    vector_init(filter->borderTab);
    vector_resize(filter->borderTab, borderLength* filter->borderElemSize);

    filter->maxWidth = filter->bufStep = 0;

    if(filter->rowBorderType == BORDER_CONSTANT || filter->columnBorderType == BORDER_CONSTANT)
    {
        filter->constBorderValue = malloc(sizeof(vector));
        vector_init(filter->constBorderValue);
        vector_resize(filter->constBorderValue, srcElemSize*borderLength);
        int srcType1 = CV_MAKETYPE(CV_MAT_DEPTH(filter->srcType), min(CV_MAT_CN(filter->srcType), 4));
        scalartoRawData(_borderValue, (unsigned char*)vector_get(filter->constBorderValue, 0), srcType1,
                        borderLength*CV_MAT_CN(filter->srcType));
    }
    filter->wholeSize = init_Point(-1, -1);
}

BaseFilter* init_BaseFilter(const Mat _kernel, Point _anchor,
        double _delta, const FilterVec_32f _vecOp)
{
    BaseFilter* f = malloc(sizeof(BaseFilter));
    f->anchor = _anchor;
    f->ksize = init_Point(_kernel.rows, _kernel.cols);
    f->delta = (float)_delta;
    f->vecOp = _vecOp;
    preprocess2DKernel(_kernel, coords, coeffs);
    vector_resize(f->ptrs, vector_size(coords));
    return f;
}

typedef struct FilterVec_32f
{
    int _nz;
    vector* coeffs; // unsigned char
    float delta;
} FilterVec_32f ;

void preprocess2DKernel(const Mat kernel, vector* coords/*Point*/, vector* coeffs/*unsigned char*/)
{
    int i, j, k, nz = countNonZero(kernel), ktype = 5;
    if(nz == 0)
        nz = 1;

    vector_resize(coords, nz);
    vector_resize(coeffs, nz*(size_t)CV_ELEM_SIZE(ktype));
    unsigned char* _coeffs = (unsigned char*)vector_get(coeffs, 0);

    for(i = k = 0; i < kernel.rows; i++)
    {
        const unsigned char* krow = ptr(kernel, i);
        for(j = 0; j < kernel.cols; j++)
        {
            float val = ((const float*)krow)[j];
            if(val == 0)
                continue;
            vector_set(coords, k, init_Point(j, i));
            ((float*)_coeffs)[k++] = val;
        }
    }
}

FilterVec_32f filterVec_32f(Mat _kernel, double _delta)
{
    FilterVec_32f f;
    f.delta = (float)_delta;
    vector* coords = malloc(sizeof(vector)); //Point
    vector_init(coords);
    f.coeffs = malloc(sizeof(vector));
    vector_init(f.coeffs);
    preprocess2DKernel(_kernel, coords, f.coeffs);
    f._nz = (int)vector_size(coords);
    return f; 
}

BaseFilter* getLinearFilter(int srcType, int dstType, Mat kernel, Point anchor,
                                double delta, int bits)
{
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(dstType);
    int cn = CV_MAT_CN(srcType), kdepth = 5;

    anchor = normalizeAnchor(&anchor, init_Point(kernel.rows, kernel.cols));
    kdepth = sdepth = 5;

    BaseFilter* f = init_BaseFilter(kernel, anchor, delta, filterVec_32f(kernel, delta));
    return f;
}

//! returns the non-separable linear filter engine
FilterEngine* createLinearFilter(int _srcType, int _dstType, Mat kernel, Point _anchor, double _detla, int _rowBorderType, int _columnBorderType, Scalar _borderValue)
{
    _srcType = CV_MAT_TYPE(_srcType);
    _dstType = CV_MAT_TYPE(_dstType);
    int cn = CV_MAT_CN(_srcType);
    assert(cn == CV_MAT_CN(_dstType));

    int bits = 0;

    BaseFilter* filter2D = getLinearFilter(_srcType, _dstType, kernel, _anchor, _delta, bits);

    BaseDimFilter *row_filter = malloc(sizeof(BaseDimFilter)), *col_filter = malloc(sizeof(BaseDimFilter));
    row_filter->ksize = row_filter->anchor = -1;
    col_filter->ksize = col_filter->anchor = -1;

    FilterEngine* filter;
    init_FilterEngine(filter, _filter2D, row_filter, col_filter, _srcType, _dstType, _srcType,
        _rowBorderType, _columnBorderType, _borderValue);

    return filter; // Check ScalartoRawData code.
}

bool isSeparable(FilterEngine f)
{
    return !f.filter2D;
}

static inline size_t alignSize(size_t sz, int n)
{
    return (sz + n-1) & -n;
}


int Start(FilterEngine* f, Point _wholeSize, Point sz, Point ofs)
{
    int i, j;

    f->wholeSize = _wholeSize;
    f->roi = init_Rect(ofs.x, ofs.y, sz.y, sz.x);

    int esz = (int)getElemSize(srcType);
    int bufElemSize = (int)getElemSize(bufType);
    const unsigned char* constVal = !vector_empty(f->constBorderValue) ? (unsigned char*)vector_get(f->constBorderValue, 0) : 0;

    int _maxBufRows = max(f->ksize.y + 3,
                               max(f->anchor.y, f->ksize.y-anchor.y-1)*2+1);

    if(f->maxWidth < f->roi.width || _maxBufRows != vector_size(f->rows))
    {
        vector_resize(f->rows, _maxBufRows);
        maxWidth = max(f->maxWidth, f->roi.width);
        int cn = CV_MAT_CN(f->srcType);
        vector_resize(f->srcRow, esz*(f->maxWidth + f->ksize.x - 1));
        if(f->columnBorderType == BORDER_CONSTANT)
        {
            assert(constVal != NULL);
            vector_resize(f->constBorderRow, (getElemSize(bufType)*(maxWidth + ksize.width - 1 + VEC_ALIGN)));
            unsigned char *dst = alignptr(vector_get(f->constBorderRow, 0), VEC_ALIGN), *tdst;
            int n = vector_size(f->constBorderValue), N;
            N = (f->maxWidth + f->ksize.x - 1)*esz;
            tdst = isSeparable(*f) ? (unsigned char*)vector_get(f->srcRow, 0) : dst;

            for(i = 0; i < N; i += n)
            {
                n = min(n, N - i);
                for(j = 0; j < n; j++)
                    tdst[i+j] = constVal[j];
            }
        }

        int maxBufStep = bufElemSize*(int)alignSize(maxWidth +
            (!isSeparable(*f) ? f->ksize.x - 1 : 0),VEC_ALIGN);
        vector_resize(f->ringBuf, maxBufStep*vector_size(f->rows)+VEC_ALIGN);
    }

    // adjust bufstep so that the used part of the ring buffer stays compact in memory
    bufStep = bufElemSize*(int)alignSize(f->roi.width + (!isSeparable(*f) ? f->ksize.x - 1 : 0),16);

    f->dx1 = max(f->anchor.x - f->roi.x, 0);
    f->dx2 = max(f->ksize.x - f->anchor.x - 1 + f->roi.x + f->roi.width - wholeSize.y, 0);

    // recompute border tables
    if(f->dx1 > 0 || f->dx2 > 0)
    {
        if(f->rowBorderType == BORDER_CONSTANT)
        {
            int nr = isSeparable(*f) ? 1 : vector_size(f->rows);
            for(i = 0; i < nr; i++)
            {
                unsigned char* dst = isSeparable(*f) ? (unsigned char*)vector_get(f->srcRow, 0) : alignptr((unsigned char*)vector_get(ringBuf, 0), VEC_ALIGN) + f->bufStep*i;
                memcpy(dst, constVal, f->dx1*esz);
                memcpy(dst + (f->roi.width + f->ksize.x - 1 - f->dx2)*esz, constVal, f->dx2*esz);
            }
        }
        else
        {
            int xofs1 = min(f->roi.x, f->anchor.x) - f->roi.x;

            int btab_esz = f->borderElemSize, wholeWidth = f->wholeSize.y;
            int* btab = (int*)vector_get(borderTab, 0);

            for(i = 0; i < f->dx1; i++)
            {
                int p0 = (borderInterpolate(i-f->dx1, wholeWidth, f->rowBorderType) + xofs1)*btab_esz;
                for(j = 0; j < btab_esz; j++)
                    btab[i*btab_esz + j] = p0 + j;
            }

            for(i = 0; i < f->dx2; i++)
            {
                int p0 = (borderInterpolate(wholeWidth + i, wholeWidth, f->rowBorderType) + xofs1)*btab_esz;
                for(j = 0; j < btab_esz; j++)
                    btab[(i + f->dx1)*btab_esz + j] = p0 + j;
            }
        }
    }

    f->rowCount = f->dstY = 0;
    f->startY = f->startY0 = max(f->roi.y - f->anchor.y, 0);
    f->endY = min(f->roi.y + f->roi.height + f->ksize.y - f->anchor.y - 1, f->wholeSize.x);
    /*
    if(f->columnFilter)
        columnFilter->reset();
    if(f->filter2D)
        filter2D->reset();
    */

    return f->startY;
}

int start(FilterEngine* f, const Mat src, const Point wsz, const Point ofs)
{
    Start(f, wsz, init_Point(src.rows, src.cols), ofs);
    return f->startY - ofs.y;
}

int remainingInputRows(FilterEngine f)
{
    return f.endY - f.startY - f.rowCount;
}

int FilterVec_32f_op(FilterVec_32f* filter, const unsigned char** _src, unsigned char* _dst, int width)
{
    const float* kf = (const float*)vector_get(filter->coeffs, 0);
    const float** src = (const float**)_src;
    float* dst = (float*)_dst;
    int i = 0, k, nz = filter->_nz;
    __m128 d4 = _mm_set1_ps(filter->delta);

    for(; i <= width - 16; i += 16)
    {
        __m128 s0 = d4, s1 = d4, s2 = d4, s3 = d4;

        for( k = 0; k < nz; k++ )
        {
            __m128 f = _mm_load_ss(kf+k), t0, t1;
            f = _mm_shuffle_ps(f, f, 0);
            const float* S = src[k] + i;

            t0 = _mm_loadu_ps(S);
            t1 = _mm_loadu_ps(S + 4);
            s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
            s1 = _mm_add_ps(s1, _mm_mul_ps(t1, f));

            t0 = _mm_loadu_ps(S + 8);
            t1 = _mm_loadu_ps(S + 12);
            s2 = _mm_add_ps(s2, _mm_mul_ps(t0, f));
            s3 = _mm_add_ps(s3, _mm_mul_ps(t1, f));
        }

        _mm_storeu_ps(dst + i, s0);
        _mm_storeu_ps(dst + i + 4, s1);
        _mm_storeu_ps(dst + i + 8, s2);
        _mm_storeu_ps(dst + i + 12, s3);
    }

    for( ; i <= width - 4; i += 4 )
    {
        __m128 s0 = d4;

        for( k = 0; k < nz; k++ )
        {
            __m128 f = _mm_load_ss(kf+k), t0;
            f = _mm_shuffle_ps(f, f, 0);
            t0 = _mm_loadu_ps(src[k] + i);
            s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
        }
        _mm_storeu_ps(dst + i, s0);
    }

    return i;
}

void filter_op(BaseFilter* f, const unsigned char** src, unsigned char* dst, int dststep, int count, int width, int cn)
{
    float _delta = f->delta;
    const Point* pt = vector_get(f->coords, 0);
    const float* kf = (const float*)vector_get(f->coeffs, 0);
    const float** kp = (const float**)(&((float*)vector_get(ptrs, 0)));
    int i, k, nz = vector_size(f->coords);

    width *= cn;
    for(; count > 0; count--, dst += dststep, src++)
    {
        float* D = (float*)dst;

        for(k = 0; k < nz; k++)
            kp[k] = (const float*)src[pt[k].y] + pt[k].x*cn;

        i = FilterVec_32f_op((const uchar**)kp, dst, width);
        for( ; i < width; i++ )
        {
            float s0 = _delta;
            for(k = 0; k < nz; k++)
                s0 += kf[k]*kp[k][i];
            D[i] = s0;
        }
    }
}

int proceed(FilterEngine* f, const unsigned char* src, int srcstep, int count,
                           unsigned char* dst, int dststep)
{
    const int *btab = (int*)vector_get(f->borderTab, 0);
    int esz = (int)getElemSize(f->srcType), btab_esz = f->borderElemSize;
    unsigned char** brows = vector_get(f->rows, 0);
    int bufRows = vector_size(f->rows);
    int cn = CV_MAT_CN(f->bufType);
    int width = f->roi.width, kwidth = f->ksize.x;
    int kheight = ksize.y, ay = f->anchor.y;
    int _dx1 = f->dx1, _dx2 = f->dx2;
    int width1 = f->roi.width + kwidth - 1;
    int xofs1 = min(f->roi.x, f->anchor.x);
    bool isSep = isSeparable(*f);
    bool makeBorder = (_dx1 > 0 || _dx2 > 0) && f->rowBorderType != BORDER_CONSTANT;
    int dy = 0, i = 0;

    src -= xofs1*esz;
    count = min(count, remainingInputRows(*f));

    assert(src && dst && count > 0);

    for(;; dst += dststep*i, dy += i)
    {
        int dcount = bufRows - ay - f->startY - f->rowCount + f->roi.y;
        dcount = dcount > 0 ? dcount : bufRows - kheight + 1;
        dcount = min(dcount, count);
        count -= dcount;

        for(; dcount-- > 0; src += srcstep)
        {
            int bi = (f->startY - f->startY0 + f->rowCount) % bufRows;
            unsigned char* brow = alignptr((unsigned char*)vector_get(ringBuf, 0), VEC_ALIGN) + bi*f->bufStep;
            unsigned char* row = isSep ? (unsigned char*)vector_get(f->srcRow, 0) : brow;

            if(++f->rowCount > bufRows)
            {
                --(f->rowCount);
                ++(f->startY);
            }

            memcpy(row + _dx1*esz, src, (width1 - _dx2 - _dx1)*esz);

            if(makeBorder)
            {
                if(btab_esz*(int)sizeof(int) == esz)
                {
                    const int* isrc = (const int*)src;
                    int* irow = (int*)row;

                    for(i = 0; i < _dx1*btab_esz; i++)
                        irow[i] = isrc[btab[i]];
                    for(i = 0; i < _dx2*btab_esz; i++)
                        irow[i + (width1 - _dx2)*btab_esz] = isrc[btab[i+_dx1*btab_esz]];
                }
                else
                {
                    for(i = 0; i < _dx1*esz; i++)
                        row[i] = src[btab[i]];
                    for(i = 0; i < _dx2*esz; i++)
                        row[i + (width1 - _dx2)*esz] = src[btab[i+_dx1*esz]];
                }
            }
        }

        int max_i = min(bufRows, f->roi.height - (f->dstY + dy) + (kheight - 1));
        for(i = 0; i < max_i; i++)
        {
            int srcY = borderInterpolate(f->dstY + dy + i + f->roi.y - ay,
                            wholeSize.height, columnBorderType);
            if(srcY < 0) // can happen only with constant border type
                brows[i] = alignptr((unsigned char*)vector_get(f->constBorderRow, 0), VEC_ALIGN);
            else
            {
                assert(srcY >= f->startY);
                if(srcY >= f->startY + f->rowCount)
                    break;
                int bi = (srcY - f->startY0) % bufRows;
                brows[i] = alignptr((unsigned char*)vector_get(ringBuf, 0), VEC_ALIGN) + bi*bufStep;
            }
        }

        if(i < kheight)
            break;
        i -= kheight - 1;
        filter_op(f->filter2D, (const uchar**)brows, dst, dststep, i, roi.width, cn);
    }

    f->dstY += dy;
    assert(f->dstY <= f->roi.height);
    return dy;
}

void apply(FilterEngine* f, Mat src, Mat* dst, Point wsz, Point ofs)
{
    int y = start(f, src, wsz, ofs);
    proceed(f, ptr(src, 0) + y*src.step,
            (int)src.step,
            f->endY - f->startY,
            ptr(*dst, 0),
            (int)dst->step);
}

static void ocvFilter2D(int stype, int dtype, int kernel_type,
                        unsigned char* src_data, size_t src_step,
                        unsigned char* dst_data, size_t dst_step,
                        int width, int height,
                        int full_width, int full_height,
                        int offset_x, int offset_y,
                        unsigned char* kernel_data, size_t kernel_step,
                        int kernel_width, int kernel_height,
                        int anchor_x, int anchor_y,
                        double delta, int borderType)
{
    int borderTypeValue = borderType & ~BORDER_ISOLATED;
    Mat kernel = createMat(kernel_height, kernel_width, kernel_type, kernel_data, kernel_step);
    FilterEngine* f = createLinearFilter(stype, dtype, kernel, init_Point(anchor_x, anchor_y), delta, borderTypeValue, -1, init_Scalar(0, 0, 0, 0));
    Mat src = createMat(height, width, stype, src_data, src_step);
    Mat dst = createMat(height, width, stype, dst_data, dst_step);
    apply(f, src, &dst, init_Point(full_height, full_width), init_Point(offset_x, offset_y));
}

void Filter2D(int stype, int dtype, int kernel_type,
              unsigned char* src_data, size_t src_step,
              unsigned char* dst_data, size_t dst_step,
              int width, int height,
              int full_width, int full_height,
              int offset_x, int offset_y,
              unsigned char* kernel_data, size_t kernel_step,
              int kernel_width, int kernel_height,
              int anchor_x, int anchor_y,
              double delta, int borderType,
              bool isSubmatrix)
{
    ocvFilter2D(stype, dtype, kernel_type,
                src_data, src_step,
                dst_data, dst_step,
                width, height,
                full_width, full_height,
                offset_x, offset_y,
                kernel_data, kernel_step,
                kernel_width, kernel_height,
                anchor_x, anchor_y,
                delta, borderType);
}


void filter2D(Mat src, Mat* dst, int ddepth, Mat kernel, Point anchor0, double delta, int borderType)
{
    if(ddepth < 0)
        ddepth = depth(src);

    create(dst, src.rows, src.cols, CV_MAKETYPE(ddepth, channels(src)));
    Point anchor = normalizeAnchor(&anchor0, init_Point(src.rows, src.cols));

    Point ofs;
    Point wsz = init_Point(src.rows, src.cols);
    if(borderType & BORDER_ISOLATED == 0)
        locateROI(src, &wsz, &ofs);

    Filter2D(5, 5, 5, 
             src.data, src.step, dst->data, dst->step,
             dst->cols, dst->rows, wsz.y, wsz.x, ofs.x, ofs.y,
             kernel.data, kernel.step, kernel.cols, kernel.rows, 
             anchor.x, anchor.y
             delta, borderType, isSubmatrix(src));
}

void magnitude(Mat X, Mat Y, Mat* Mag)
{
    int type = type(X), depth = depth(X), cn = channels(X);
    create(Mag, X.rows, X.cols, type);

    const Mat* arrays[] = {&X, &Y, &Mag, 0};
    unsigned char* ptrs[3];
    MatIterator it = matIterator(arrays, 0, ptrs, 3);
    int len = (int)it.size*cn;

    for(size_t i = 0; i < it.nplanes; i++, getNextIterator(&it))
    {
        const float *x = (const float*)ptrs[0], *y = (const float*)ptrs[1];
        float *mag = (float*)ptrs[2];
        for(int j = 0; j < len; j++)
        {
            double x0 = x[j], y0 = y[j];
            mag[j] = sqrt(x0*x0 + y0*y0);
        }
    }
}

void get_gradient_magnitude(Mat* _grey_image, Mat* _gradient_magnitude)
{
    Mat C = Mat_<float>(_grey_img);

    Mat kernel;
    create(&kernel, 1, 3, 5);
    float sample[3] = {-1, 0, 1};
    leftshift_op(&kernel, 3, sample);

    Mat grad_x;
    filter2D(C, grad_x, -1, kernel, init_Point(-1,-1), 0, BORDER_DEFAULT);

    Mat kernel2;
    create(&kernel2, 3, 1, 5);
    leftshift_op(&kernel, 3, sample);
    Mat grad_y;
    filter2D(C, grad_y, -1, kernel2, init_Point(-1,-1), 0, BORDER_DEFAULT);

    magnitude(grad_x, grad_y, _gradient_magnitude);
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
    region_pair pair;
    pair.a = a;
    pair.b = b;
    return pair;
}

region_triplet init_region_triplet(Point _a, Point _b, Point _c)
{
    region_triplet triplet;
    triplet.a = _a;
    triplet.b = _b;
    triplet.c = _c;
    return triplet;
}


void convexHull(Mat points, vector* hull, bool clockwise /* false */, bool returnPoints/* true */)
{
    int i, total = checkVector(points, 2, -1, true), depth = depth(points), nout = 0;
    int miny_ind = 0, maxy_ind = 0;
    assert(total >= 0 && (depth == CV_32F || depth == CV_32S));

    if(total == 0)
    {
        vector_free(hull);
        return;
    }

    bool is_float = depth == CV_32F;
    Point** pointer = malloc(sizeof(Point*) * total);
    floatPoint** pointerf = (floatPoint**)pointer;
    Point* data0 = (Point*)ptr(points, 0);
    int* stack = malloc(sizeof(total+2));
    int* hullbuf = malloc(sizeof(total));

    assert(isContinuous(points));

    for(i = 0; i < total; i++)
        pointer[i] = &data0[i];

    // sort the point set by x-coordinate, find min and max y
    if(!is_float)
    {
        qsort(pointer, total, sizeof(Point*), CHullCmpPoints<int>());
        for(i = 1; i < total; i++)
        {
            int y = pointer[i]->y;
            if(pointer[miny_ind]->y > y)
                miny_ind = i;
            if(pointer[maxy_ind]->y < y)
                maxy_ind = i;
        }
    }
    
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

MatIterator matIterator(const Mat** _arrays, Mat* _planes, unsigned char** _ptrs, int _narrays)
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

typedef struct RGB2HLS_f
{
    int srccn, blueIdx;
    float hscale;
} RGB2HLS_f ;

void RGBtoHLS_f(RGB2HLS_f* body, const float* src, float* dst, int n)
{
    int i = 0, bidx = body->blueIdx, scn = body->srccn;
    n *= 3;

    for(; i < n; i += 3, src += scn)
    {
        float b = src[bidx], g = src[1], r = src[bidx^2];
        float h = 0.f, s = 0.f, l;
        float vmin, vmax, diff;

        vmax = vmin = r;
        if(vmax < g) vmax = g;
        if(vmax < b) vmax = b;
        if(vmin > g) vmin = g;
        if(vmin > b) vmin = b;

        diff = vmax - vmin;
        l = (vmax + vmin)*0.5f;

        if(diff > FLT_EPSILON)
        {
            s = l < 0.5f ? diff/(vmax + vmin) : diff/(2 - vmax - vmin);
            diff = 60.f/diff;

            if(vmax == r)
                h = (g-b)*diff;
            else if(vmax == g)
                h = (b-r)*diff + 120.f;
            else
                h = (r-g)*diff + 240.f;

            if(h < 0.f) h += 360.f;
        }

        dst[i] = h*hscale;
        dst[i+1] = l;
        dst[i+2] = s;
    }
}

typedef struct RGB2HLS_b
{
    int srccn;
    RGB2HLS_f cvt;
} RGB2HLS_b ;

RGB2HLS_b RGB2HLSb(int _srccn, int _blueIdx, int _hrange)
{
    RGB2HLS_b rgb2hls;
    rgb2hls.srccn = _srccn;
    rgb2hls.cvt.srccn = 3;
    rgb2hls.cvt.blueIdx = _blueIdx;
    rgb2hls.cvt.hscale = (float)_hrange;
    return rgb2hls;
}

void cvtBGRtoLab(const unsigned char * src_data, size_t src_step,
                 unsigned char* dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isLab, bool srgb)
{
    int blueIdx = swapBlue ? 2 : 0;

    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, RGB2Lab_b(scn, blueIdx, 0, 0, srgb), COLOR_RGB2Lab);
}

void cvtColorBGR2Lab(Mat _src, Mat* _dst, bool swapb, bool srgb)
{
    CvtHelper h = cvtHelper(_src, _dst, 3);

    cvtBGRtoLab(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                     h.depth, h.scn, swapb, true, srgb);
}

void cvtColor(Mat src, Mat* dst, int code)
{
    if(code == COLOR_RGB2GRAY)
        cvtColorBGR2Gray(src, dst, true);

    if(code == COLOR_RGB2HLS)
        cvtColorBGR2HLS(src, dst, true, false);

    if(code == COLOR_RGB2Lab)
        cvtColorBGR2Lab(src, dst, true, true);
}

void cvtBGRtoHSV(const unsigned char * src_data, size_t src_step,
                 unsigned char * dst_data, size_t dst_step,
                 int width, int height,
                 int depth, int scn, bool swapBlue, bool isFullRange, bool isHSV)
{
    int hrange = depth == CV_32F ? 360 : isFullRange ? 256 : 180;
    int blueIdx = swapBlue ? 2 : 0;
    RGB2HLS_b r2h = RGB2HLSb(scn, blueIdx, hrange);
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, &r2h, COLOR_RGB2HLS);
}


void cvtColorBGR2HLS(Mat _src, Mat* _dst, bool swapb, bool fullRange)
{
    CvtHelper h = cvtHelper(_src, _dst, 3);

    cvtBGRtoHSV(h.src.data, h.src.step, h.dst.data, h.dst.step, h.src.cols, h.src.rows,
                h.depth, h.scn, swapb, fullRange, false);
}


void RGB2Hls_b(RGB2HLS_b* r2h, const unsigned char* src, unsigned char* dst, int n)
{
    int i, j, scn = r2h->srccn;
    float buf[3*BLOCK_SIZE];

    for(i = 0; i < n; i += BLOCK_SIZE, dst += BLOCK_SIZE*3)
    {
        int dn = min(n-i, (int)BLOCK_SIZE);
        j = 0;

        for(; j < dn*3; j += 3, src += scn)
        {
            buf[j] = src[0]*(1.f/255.f);
            buf[j+1] = src[1]*(1.f/255.f);
            buf[j+2] = src[2]*(1.f/255.f);
        }
        RGBtoHLS_f(&(r2h->cvt), buf, buf, dn);

        j = 0;
        for(; j < dn*3; j += 3)
        {
            dst[j] = (unsigned char)(buf[j]);
            dst[j+1] = (unsigned char)(buf[j+1]*255.f);
            dst[j+2] = (unsigned char)(buf[j+2]*255.f);
        }
    }
}


void cvtColorBGR2Gray(Mat _src, Mat *_dst, bool swapb)
{
    CvtHelper h = cvtHelper(_src, _dst, 1);
    CvtBGRtoGray(h.src.data, h.src.step[0], h.dst.data, h.dst.step[0], h.src.cols, h.src.rows,
                      h.depth, h.scn, swapb);
}

void CvtBGRtoGray(unsigned char* src_data, size_t src_step,
                             unsigned char * dst_data, size_t dst_step,
                             int width, int height,
                             int depth, int scn, bool swapBlue);
{
    int blueIdx = swapBlue ? 2 : 0;
    RGBtoGray r2g = RGB2Gray(scn, blueIdx, 0);
    CvtColorLoop(src_data, src_step, dst_data, dst_step, width, height, &r2g, COLOR_RGB2GRAY);
}

void RGB2gray(RGBtoGray* cvt, unsigned char* src, unsigned char* dst, int n)
{
    int scn = cvt->srccn;
    const int* _tab = cvt->tab;
    for(int i = 0; i < n; i++, src += scn)
        dst[i] = (unsigned char)((_tab[src[0]] + _tab[src[1]+256] + _tab[src[2]+512]) >> 14);
}

typedef struct  RGB2Lab_b
{
    int srccn;
    int coeffs[9];
    bool srgb;
} RGB2Lab_b ;

void RGB2lab_b(RGB2Lab_b* r2l, int _srccn, int blueIdx, const float* _coeffs,
              const float* _whitept, bool _srgb)
{
    r2l->srccn = _srccn;
    r2l->srgb = _srgb;
    static volatile int _3 = 3;

}

void rangeop_(CvtColorLoop_Invoker* body, Point range, int code)
{
    const unsigned char* yS = body->src_data + (size_t)(range.x)*body->src_step;
    unsigned char* yD = body->dst_data + (size_t)(range.x)*body->dst_step;

    for(int i = range.x; i < range.y; ++i, yS += src_step, yD += dst_step)
    {
        if(code == COLOR_RGB2GRAY)
            RGB2gray((RGBtoGray*)(body->cvt), yS, yD, width);

        if(code == COLOR_RGB2HLS)
            RGB2Hls_b((RGB2HLS_b*)(body->cvt), yS, yD, width);

        if(code == COLOR_RGB2Lab)
            RGB2lab_b((RGB2Lab_b*)(body->cvt), yS, yD, width);
    }
}

void parallel_for__(Point range, CvtColorLoop_Invoker body, double nstripes, int code)
{
    if(range.x == range.y)
        return;
    (void)nstripes;
    rangeop_(&body, range, code);
}

void CvtColorLoop(const unsigned char* src_data, size_t src_step, unsigned char* dst_data, size_t dst_step, int width, int height, void* cvt, int code)
{
    parallel_for__(init_Point(0, height),
                  cvtColorLoop_Invoker(src_data, src_step, dst_data, dst_step, width, cvt),
                  (width * height) / (double)(1<<16), code);
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
                   vector* buffer) //<unsigned char, unsigned char, int, Diff8uC1>
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
                   vector* buffer) //<Vec3b, unsigned char, Vec3i, Diff8uC3>
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

static int icvTraceContour_32s(int *ptr, int step, int *stop_ptr, int is_hole)
{
    assert(ptr != NULL);
    int deltas[MAX_SIZE];
    int *i0 = ptr, *i1, *i3, *i4 = NULL;
    int s, s_end;
    const int   right_flag = INT_MIN;
    const int   new_flag = (int)((unsigned)INT_MIN >> 1);
    const int   value_mask = ~(right_flag | new_flag);
    const int   ccomp_val = *i0 & value_mask;

    /* initialize local state */
    CV_INIT_3X3_DELTAS(deltas, step, 1);
    memcpy(deltas + 8, deltas, 8 * sizeof( deltas[0]));

    s_end = s = is_hole ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while((*i1 & value_mask) != ccomp_val && s != s_end);

    i3 = i0;

    /* check single pixel domain */
    if(s != s_end)
    {
        /* follow border */
        for(;;)
        {
            assert(i3 != NULL);
            s_end = s;
            s = min(s, MAX_SIZE - 1);

            while(s < MAX_SIZE - 1)
            {
                i4 = i3 + deltas[++s];
                assert(i4 != NULL);
                if((*i4 & value_mask) == ccomp_val)
                    break;
            }

            if(i3 == stop_ptr || (i4 == i0 && i3 == i1))
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }
    return i3 == stop_ptr;
}

/*
   trace contour until certain point is met.
   returns 1 if met, 0 else.
*/
static int icvTraceContour(signed char *ptr, int step, signed char *stop_ptr, int is_hole)
{
    int deltas[MAX_SIZE];
    signed char *i0 = ptr, *i1, *i3, *i4 = NULL;
    int s, s_end;

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy(deltas + 8, deltas, 8 * sizeof(deltas[0]));

    s_end = s = is_hole ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while(*i1 == 0 && s != s_end);

    i3 = i0;

    /* check single pixel domain */
    if(s != s_end)
    {
        /* follow border */
        for(;;)
        {
            assert(i3 != NULL);

            s = std::min(s, MAX_SIZE - 1);
            while( s < MAX_SIZE - 1 )
            {
                i4 = i3 + deltas[++s];
                assert(i4 != NULL);
                if(*i4 != 0)
                    break;
            }

            if(i3 == stop_ptr || (i4 == i0 && i3 == i1))
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }
    return i3 == stop_ptr;
}

/* Initialize sequence writer: */
void StartAppendToSeq(Seq *seq, SeqWriter * writer)
{
    if(!seq || !writer)
        fatal("NULL Pointer Error");

    memset(writer, 0, sizeof(*writer));
    writer->header_size = sizeof(SeqWriter);

    writer->seq = seq;
    writer->block = seq->first ? seq->first->prev : 0;
    writer->ptr = seq->ptr;
    writer->block_max = seq->block_max;
}

/* Update sequence header: */
void cvFlushSeqWriter(SeqWriter* writer)
{
    if(!writer)
        fatal("NULL Pointer Error");

    Seq* seq = writer->seq;
    seq->ptr = writer->ptr;

    if(writer->block)
    {
        int total = 0;
        SeqBlock *first_block = writer->seq->first;
        SeqBlock *block = first_block;

        writer->block->count = (int)((writer->ptr - writer->block->data) / seq->elem_size);
        assert( writer->block->count > 0 );

        do
        {
            total += block->count;
            block = block->next;
        }
        while( block != first_block );

        writer->seq->total = total;
    }
}


/* Calls icvFlushSeqWriter and finishes writing process: */
Seq* EndWriteSeq(SeqWriter* writer)
{
    if(!writer)
        fatal("NULL Pointer Error");

    cvFlushSeqWriter(writer);
    Seq* seq = writer->seq;

    /* Truncate the last block: */
    if(writer->block && writer->seq->storage)
    {
        MemStorage *storage = seq->storage;
        signed char *storage_block_max = (signed char *) storage->top + storage->block_size;

        assert(writer->block->count > 0);

        if((unsigned)((storage_block_max - storage->free_space)
            - seq->block_max) < CV_STRUCT_ALIGN )
        {
            storage->free_space = AlignLeft((int)(storage_block_max - seq->ptr), CV_STRUCT_ALIGN);
            seq->block_max = seq->ptr;
        }
    }

    writer->ptr = 0;
    return seq;
}

Mat* GetMat(const void* array, Mat* mat, int* pCOI, int allowND)
{
    Mat* result = 0;
    Mat* src = (Mat*)array;

    if(!mat || !src)
        fatal("NULL array pointer is passed");

    if(!src->data)
        fatal("The matrix has NULL data pointer");

    result = (Mat*)src;

    if( pCOI )
        *pCOI = coi;

    return result;
}

Mat* cvReshape(const void* array, Mat* header,
           int new_cn, int new_rows)
{
    Mat* result = 0;
    Mat *mat = (Mat*)array;
    int total_width, new_width;

    if(!header)
        fatal("NULL Pointer Error");

    if(!CV_IS_MAT(mat))
    {
        int coi = 0;
        mat = GetMat(mat, header, &coi, 1);
        if(coi)
            fatal("COI is not supported");
    }

    if( new_cn == 0 )
        new_cn = CV_MAT_CN(mat->type);
    else if((unsigned)(new_cn - 1) > 3)
        fatal("Bad number of channels");

    if(mat != header)
    {
        int hdr_refcount = header->hdr_refcount;
        *header = *mat;
        header->refcount = 0;
        header->hdr_refcount = hdr_refcount;
    }

    total_width = mat->cols * CV_MAT_CN(mat->type);

    if((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0 )
        new_rows = mat->rows * total_width / new_cn;

    if(new_rows == 0 || new_rows == mat->rows)
    {
        header->rows = mat->rows;
        header->step = mat->step;
    }
    else
    {
        int total_size = total_width * mat->rows;
        if( !CV_IS_MAT_CONT(mat->type))
            fatal("The matrix is not continuous, thus its number of rows can not be changed");

        if((unsigned)new_rows > (unsigned)total_size)
            fatal("Bad new number of rows");

        total_width = total_size / new_rows;

        if(total_width * new_rows != total_size)
            fatal("The total number of matrix elements "
                                    "is not divisible by the new number of rows");

        header->rows = new_rows;
        header->step = total_width * CV_ELEM_SIZE1(mat->type);
    }

    new_width = total_width / new_cn;

    if(new_width * new_cn != total_width)
        fatal("The total width is not divisible by the new number of channels" );

    header->cols = new_width;
    header->type = (mat->type  & ~CV_MAT_TYPE_MASK) | CV_MAKETYPE(mat->type, new_cn);

    result = header;
    return  result;
}

/* Construct a sequence from an array without copying any data.
   NB: The resultant sequence cannot grow beyond its initial size: */
Seq* MakeSeqHeaderForArray(int seq_flags, int header_size, int elem_size,
                         void *array, int total, Seq *seq, SeqBlock * block)
{
    Seq* result = 0;

    if(elem_size <= 0 || header_size < (int)sizeof(Seq) || total < 0)
        fatal("Bad size error");

    if(!seq || ((!array || !block) && total > 0))
        fatal("NULL Pointer Error");

    memset(seq, 0, header_size);

    seq->header_size = header_size;
    seq->flags = (seq_flags & ~CV_MAGIC_MASK) | CV_SEQ_MAGIC_VAL;
    {
        int elemtype = CV_MAT_TYPE(seq_flags);
        int typesize = CV_ELEM_SIZE(elemtype);

        if( elemtype != CV_SEQ_ELTYPE_GENERIC &&
            typesize != 0 && typesize != elem_size )
            fatal("Element size doesn't match to the size of predefined element type "
            "(try to use 0 for sequence element type)");
    }
    seq->elem_size = elem_size;
    seq->total = total;
    seq->block_max = seq->ptr = (schar *) array + total * elem_size;

    if(total > 0)
    {
        seq->first = block;
        block->prev = block->next = block;
        block->start_index = 0;
        block->count = total;
        block->data = (schar *) array;
    }

    result = seq;

    return result;
}


Seq* cvPointSeqFromMat(int seq_kind, const void* arr,
                                  Contour* contour_header, SeqBlock* block)
{
    assert(arr != 0 && contour_header != 0 && block != 0);

    int eltype;
    Mat hdr;
    Mat* mat = (Mat*)arr;

    if(!CV_IS_MAT(mat))
        fatal("Imput array is not a valid matrix");

    if( CV_MAT_CN(mat->type) == 1 && mat->width == 2 )
        mat = cvReshape(mat, &hdr, 2);

    eltype = CV_MAT_TYPE(mat->type);
    if(eltype != CV_32SC2 && eltype != CV_32FC2)
        fatal("The matrix can not be converted to point sequence because of 
        inappropriate element type");

    if( (mat->width != 1 && mat->height != 1) || !CV_IS_MAT_CONT(mat->type))
        fatal("The matrix converted to point sequence must be "
        "1-dimensional and continuous");

    MakeSeqHeaderForArray((seq_kind & (CV_SEQ_KIND_MASK|CV_SEQ_FLAG_CLOSED)) | eltype,
            sizeof(Contour), CV_ELEM_SIZE(eltype), mat->data,
            mat->width*mat->height, (Seq*)contour_header, block);

    return (Seq*)contour_header;
}

static Rect maskBoundingRect(Mat* img)
{
    assert(depth(*img) <= CV_8S && channels(*img) == 1);

    int size[2] = {src->rows, src->cols};
    int xmin = size[1], ymin = -1, xmax = -1, ymax = -1, i, j, k;

    for(i = 0; i < size[0]; i++)
    {
        const unsigned char* _ptr = ptr(*img, i);
        const unsigned char* ptr = (const unsigned char*)alignptr(_ptr, 4);
        int have_nz = 0, k_min, offset = (int)(ptr - _ptr);
        j = 0;
        offset = min(offset, size[1]);
        for(; j < offset; j++)
            if(_ptr[j])
            {
                have_nz = 1;
                break;
            }
        if(j < offset)
        {
            if(j < xmin)
                xmin = j;
            if(j > xmax)
                xmax = j;
        }
        if(offset < size[1])
        {
            xmin -= offset;
            xmax -= offset;
            size.width -= offset;
            j = 0;
            for(; j <= xmin - 4; j += 4)
                if(*((int*)(ptr+j)))
                    break;
            for(; j < xmin; j++)
                if(ptr[j])
                {
                    xmin = j;
                    if(j > xmax)
                        xmax = j;
                    have_nz = 1;
                    break;
                }
            k_min = max(j-1, xmax);
            k = size[1] - 1;
            for(; k > k_min && (k&3) != 3; k--)
                if(ptr[k])
                    break;
            if(k > k_min && (k&3) == 3)
            {
                for(; k > k_min+3; k -= 4)
                    if(*((int*)(ptr+k-3)))
                        break;
            }
            for(; k > k_min; k--)
                if(ptr[k])
                {
                    xmax = k;
                    have_nz = 1;
                    break;
                }
            if(!have_nz)
            {
                j &= ~3;
                for(; j <= k - 3; j += 4)
                    if(*((int*)(ptr+j)))
                        break;
                for(; j <= k; j++)
                    if(ptr[j])
                    {
                        have_nz = 1;
                        break;
                    }
            }
            xmin += offset;
            xmax += offset;
            size.width += offset;
        }
        if(have_nz)
        {
            if( ymin < 0 )
                ymin = i;
            ymax = i;
        }
    }

    if(xmin >= size.width)
        xmin = ymin = 0;
    return init_Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

int checkVector(Mat m, int _elemChannels, int _depth /* -1 */, bool _requireContinuous/* true */)
{
    return m.data && (depth(m) == _depth || _depth <= 0) && (isContinuous(m) || !_requireContinuous) &&
    (((m.rows == 1 || m.cols == 1) && channels(m) == _elemChannels) ||
                        (m.cols == _elemChannels && channels(m) == 1))?(total(m)*channels(m)/_elemChannels) : -1;
}

// Calculates bounding rectagnle of a point set or retrieves already calculated
static Rect pointSetBoundingRect(Mat points)
{
    int npoints = checkVector(points, 2, -1, true);
    int depth = depth(points);
    assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i;
    bool is_float = depth == CV_32F;

    if(npoints == 0)
        return init_Rect(0, 0, 0, 0);

    const Point* pts = (Point*)ptr(points, 0);
    Point pt = pts[0];

    if(!is_float)
    {
        xmin = xmax = pt.x;
        ymin = ymax = pt.y;

        for(i = 1; i < npoints; i++)
        {
            pt = pts[i];

            if(xmin > pt.x)
                xmin = pt.x;

            if(xmax < pt.x)
                xmax = pt.x;

            if(ymin > pt.y)
                ymin = pt.y;

            if(ymax < pt.y)
                ymax = pt.y;
        }
    }
    else
    {

            Cv32suf v;
            // init values
            xmin = xmax = CV_TOGGLE_FLT(pt.x);
            ymin = ymax = CV_TOGGLE_FLT(pt.y);

            for(i = 1; i < npoints; i++)
            {
                pt = pts[i];
                pt.x = CV_TOGGLE_FLT(pt.x);
                pt.y = CV_TOGGLE_FLT(pt.y);

                if(xmin > pt.x)
                    xmin = pt.x;

                if(xmax < pt.x)
                    xmax = pt.x;

                if(ymin > pt.y)
                    ymin = pt.y;

                if(ymax < pt.y)
                    ymax = pt.y;
            }

            v.i = CV_TOGGLE_FLT(xmin); xmin = Floor(v.f);
            v.i = CV_TOGGLE_FLT(ymin); ymin = Floor(v.f);
            // because right and bottom sides of the bounding rectangle are not inclusive
            // (note +1 in width and height calculation below), cvFloor is used here instead of cvCeil
            v.i = CV_TOGGLE_FLT(xmax); xmax = Floor(v.f);
            v.i = CV_TOGGLE_FLT(ymax); ymax = Floor(v.f);
    }

    return init_Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}


/* Calculates bounding rectangle of a point set or retrieves already calculated */
Rect cvBoundingRect(void* array, int update)
{
    Rect rect;
    Contour contour_header;
    Seq* ptseq = 0;
    SeqBlock block;

    Mat stub, *mat = 0;
    int calculate = update;

    if(CV_IS_SEQ(array))
    {
        ptseq = (Seq*)array;
        if(!CV_IS_SEQ_POINT_SET(ptseq))
            fatal("Unsupported sequence type" );

        if(ptseq->header_size < (int)sizeof(Contour))
        {
            update = 0;
            calculate = 1;
        }
    }
    else
    {
        mat = GetMat(array, &stub);
        if(CV_MAT_TYPE(mat->type) == CV_32SC2 ||
            CV_MAT_TYPE(mat->type) == CV_32FC2)
        {
            ptseq = cvPointSeqFromMat(CV_SEQ_KIND_GENERIC, mat, &contour_header, &block);
            mat = 0;
        }
        else if( CV_MAT_TYPE(mat->type) != CV_8UC1 &&
                CV_MAT_TYPE(mat->type) != CV_8SC1 )
            fatal("The image/matrix format is not supported by the function");
        update = 0;
        calculate = 1;
    }

    if(!calculate)
        return ((Contour*)ptseq)->rect;

    if(mat)
    {
        rect = maskBoundingRect(mat);
    }
    else if(ptseq->total)
    {
        AutoBuffer abuf;
        rect = pointSetBoundingRect(cv::cvarrToMat(ptseq, false, false, 0, &abuf));
    }
    if(update)
        ((Contour*)ptseq)->rect = rect;
    return rect;
}

static void icvFetchContour(signed char *ptr, int step, Point pt, Seq* contour, int _method)
{
    const signed char     nbd = 2;
    int deltas[MAX_SIZE];
    SeqWriter writer;
    signed char *i0 = ptr, *i1, *i3, *i4 = 0;
    int prev_s = -1, s, s_end;
    int method = _method - 1;

    /* initialize local state */
    CV_INIT_3X3_DELTAS(deltas, step, 1+);
    memcpy(deltas + 8, deltas, 8 * sizeof(deltas[0]));

    /* initialize writer */
    StartAppendToSeq(contour, &writer);

    if(method < 0)
        ((Chain*)contour)->origin = pt;

    s_end = s = CV_IS_SEQ_HOLE(contour) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while(*i1 == 0 && s != s_end);

    if(s == s_end)            /* single pixel domain */
    {
        *i0 = (signed char)(nbd | -128);
        if(method >= 0)
        {
            CV_WRITE_SEQ_ELEM(pt, writer);
        }
    }
    else
    {
        i3 = i0;
        prev_s = s ^ 4;

        /* follow border */
        for(;;)
        {
            assert(i3 != NULL);
            s_end = s;
            s = min(s, MAX_SIZE - 1);

            while(s < MAX_SIZE - 1)
            {
                i4 = i3 + deltas[++s];
                assert(i4 != NULL);
                if(*i4 != 0)
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if((unsigned)(s-1) < (unsigned)s_end)
            {
                *i3 = (signed char)(nbd | -128);
            }
            else if(*i3 == 1)
            {
                *i3 = nbd;
            }

            if(method < 0)
            {
                signed char _s = (signed char) s;

                CV_WRITE_SEQ_ELEM(_s, writer);
            }
            else
            {
                if(s != prev_s || method == 0)
                {
                    CV_WRITE_SEQ_ELEM( pt, writer );
                    prev_s = s;
                }

                pt.x += icvCodeDeltas[s].x;
                pt.y += icvCodeDeltas[s].y;

            }

            if( i4 == i0 && i3 == i1 )
                break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    EndWriteSeq(&writer);

    if(_method != CV_CHAIN_CODE)
        cvBoundingRect(contour, 1);

}

/* Add new element to the set: */
int SetAdd(Set* set, SetElem* element, SetElem** inserted_element)
{
    int id = -1;
    SetElem *free_elem;

    if(!set)
        fatal("NULL Pointer Error");

    if(!(set->free_elems))
    {
        int count = set->total;
        int elem_size = set->elem_size;
        schar *ptr;
        icvGrowSeq((Seq *)set, 0 );

        set->free_elems = (SetElem*)(ptr = set->ptr);
        for( ; ptr + elem_size <= set->block_max; ptr += elem_size, count++ )
        {
            ((SetElem*)ptr)->flags = count | CV_SET_ELEM_FREE_FLAG;
            ((SetElem*)ptr)->next_free = (SetElem*)(ptr + elem_size);
        }
        assert( count <= CV_SET_ELEM_IDX_MASK+1 );
        ((SetElem*)(ptr - elem_size))->next_free = 0;
        set->first->prev->count += count - set->total;
        set->total = count;
        set->ptr = set->block_max;
    }

    free_elem = set->free_elems;
    set->free_elems = free_elem->next_free;

    id = free_elem->flags & CV_SET_ELEM_IDX_MASK;
    if(element)
        memcpy(free_elem, element, set->elem_size);

    free_elem->flags = id;
    set->active_count++;

    if(inserted_element)
        *inserted_element = free_elem;

    return id;
}

static void icvFetchContourEx_32s(int* ptr, int step, Point pt, Seq* contour, int _method, Rect* _rect)
{
    assert(ptr != NULL);
    int deltas[MAX_SIZE];
    SeqWriter writer;
    int *i0 = ptr, *i1, *i3, *i4;
    Rect rect;
    int prev_s = -1, s, s_end;
    int method = _method-1;
    const int right_flag = INT_MIN;
    const int new_flag = (int)((unsigned)INT_MIN >> 1);
    const int value_mask = ~(right_flag | new_flag);
    const int ccomp_val = *i0 & value_mask;
    const int nbd0 = ccomp_val | new_flag;
    const int nbd1 = nbd0 | right_flag;

    /* initialize local state */
    CV_INIT_3X3_DELTAS(deltas, step, 1);
    memcpy(deltas + 8, deltas, 8 * sizeof(deltas[0]));

    /* initialize writer */
    StartAppendToSeq(contour, &writer);

    if(method < 0)
        ((Chain*)contour)->origin = pt;

    rect.x = rect.width = pt.x;
    rect.y = rect.height = pt.y;

    s_end = s = CV_IS_SEQ_HOLE(contour) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while((*i1 & value_mask) != ccomp_val && s != s_end && (s < MAX_SIZE - 1));

    if(s == s_end)            /* single pixel domain */
    {
        *i0 = nbd1;
        if(method >= 0)
        {
            CV_WRITE_SEQ_ELEM(pt, writer);
        }
    }
    else
    {
        i3 = i0;
        prev_s = s ^ 4;

        /* follow border */
        for(;;)
        {
            assert(i3 != NULL);
            s_end = s;

            do
            {
                i4 = i3 + deltas[++s];
                assert(i4 != NULL);
            }
            while((*i4 & value_mask) != ccomp_val && (s < MAX_SIZE - 1));
            s &= 7;

            /* check "right" bound */
            if((unsigned)(s-1) < (unsigned) s_end)
            {
                *i3 = nbd1;
            }
            else if(*i3 == ccomp_val)
            {
                *i3 = nbd0;
            }

            if(method < 0)
            {
                signed char _s = (signed char)s;
                CV_WRITE_SEQ_ELEM( _s, writer );
            }
            else if(s != prev_s || method == 0)
            {
                CV_WRITE_SEQ_ELEM(pt, writer);
            }

            if(s != prev_s)
            {
                /* update bounds */
                if(pt.x < rect.x)
                    rect.x = pt.x;
                else if(pt.x > rect.width)
                    rect.width = pt.x;

                if(pt.y < rect.y)
                    rect.y = pt.y;
                else if(pt.y > rect.height)
                    rect.height = pt.y;
            }

            prev_s = s;
            pt.x += icvCodeDeltas[s].x;
            pt.y += icvCodeDeltas[s].y;

            if( i4 == i0 && i3 == i1 )  break;

            i3 = i4;
            s = (s + 4) & 7;
        }                       /* end of border following loop */
    }

    rect.width -= rect.x - 1;
    rect.height -= rect.y - 1;

    EndWriteSeq(&writer);

    if(_method != CV_CHAIN_CODE)
        ((Contour*)contour)->rect = rect;

    if( _rect )  *_rect = rect;
}

static void icvFetchContourEx(signed char* ptr, int step, Point pt, Seq* contour, int  _method, int nbd, Rect* _rect)
{
    int deltas[MAX_SIZE];
    SeqWriter writer;
    signed char *i0 = ptr, *i1, *i3, *i4 = NULL;
    Rect rect;
    int prev_s = -1, s, s_end;
    int method = _method - 1;

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1);
    memcpy(deltas + 8, deltas, 8 * sizeof(deltas[0]));

    /* initialize writer */
    StartAppendToSeq(contour, &writer);

    if(method < 0)
        ((Chain *)contour)->origin = pt;

    rect.x = rect.width = pt.x;
    rect.y = rect.height = pt.y;

    s_end = s = CV_IS_SEQ_HOLE( contour ) ? 0 : 4;

    do
    {
        s = (s - 1) & 7;
        i1 = i0 + deltas[s];
    }
    while(*i1 == 0 && s != s_end);

    if(s == s_end)            /* single pixel domain */
    {
        *i0 = (signed char)(nbd | 0x80);
        if(method >= 0)
        {
            CV_WRITE_SEQ_ELEM(pt, writer);
        }
    }
    else
    {
        i3 = i0;

        prev_s = s ^ 4;

        /* follow border */
        for(;;)
        {
            assert(i3 != NULL);
            s_end = s;
            s = min(s, MAX_SIZE-1);

            while(s < MAX_SIZE - 1)
            {
                i4 = i3 + deltas[++s];
                assert(i4 != NULL);
                if(*i4 != 0)
                    break;
            }
            s &= 7;

            /* check "right" bound */
            if((unsigned)(s-1) < (unsigned) s_end)
            {
                *i3 = (signed char)(nbd | 0x80);
            }
            else if(*i3 == 1)
            {
                *i3 = (signed char) nbd;
            }

            if(method < 0)
            {
                schar _s = (signed char) s;
                CV_WRITE_SEQ_ELEM(_s, writer);
            }
            else if( s != prev_s || method == 0 )
            {
                CV_WRITE_SEQ_ELEM( pt, writer );
            }

            if(s != prev_s)
            {
                /* update bounds */
                if(pt.x < rect.x)
                    rect.x = pt.x;
                else if(pt.x > rect.width)
                    rect.width = pt.x;

                if(pt.y < rect.y)
                    rect.y = pt.y;
                else if(pt.y > rect.height)
                    rect.height = pt.y;
            }

            prev_s = s;
            pt.x += icvCodeDeltas[s].x;
            pt.y += icvCodeDeltas[s].y;

            if(i4 == i0 && i3 == i1)  break;

            i3 = i4;
            s = (s+4) & 7;
        }                       /* end of border following loop */
    }

    rect.width -= rect.x - 1;
    rect.height -= rect.y - 1;

    EndWriteSeq(&writer);

    if(_method != CV_CHAIN_CODE)
        ((Contour*)contour)->rect = rect;

    if(_rect)  *_rect = rect;
}

/* Initialize sequence reader: */
void StartReadSeq(const Seq *seq, SeqReader* reader, int reverse)
{
    SeqBlock *first_block;
    SeqBlock *last_block;

    if(reader)
    {
        reader->seq = 0;
        reader->block = 0;
        reader->ptr = reader->block_max = reader->block_min = 0;
    }

    if(!seq || !reader)
        fatal("NULL Pointer Error");

    reader->header_size = sizeof(SeqReader);
    reader->seq = (Seq*)seq;

    first_block = seq->first;

    if(first_block)
    {
        last_block = first_block->prev;
        reader->ptr = first_block->data;
        reader->prev_elem = CV_GET_LAST_ELEM(seq, last_block);
        reader->delta_index = seq->first->start_index;

        if(reverse)
        {
            signed char *temp = reader->ptr;

            reader->ptr = reader->prev_elem;
            reader->prev_elem = temp;

            reader->block = last_block;
        }
        else
        {
            reader->block = first_block;
        }

        reader->block_min = reader->block->data;
        reader->block_max = reader->block_min + reader->block->count * seq->elem_size;
    }
    else
    {
        reader->delta_index = 0;
        reader->block = 0;

        reader->ptr = reader->prev_elem = reader->block_min = reader->block_max = 0;
    }
}

void StartReadChainPoints(Chain * chain, ChainPtReader* reader)
{
    int i;

    if(!chain || !reader)
        fatal("NULL Pointer Error");

    if(chain->elem_size != 1 || chain->header_size < (int)sizeof(CvChain))
        fatal("Bad size error");

    StartReadSeq((Seq*)chain, (SeqReader*)reader, 0);

    reader->pt = chain->origin;
    for( i = 0; i < 8; i++ )
    {
        reader->deltas[i][0] = (schar) icvCodeDeltas[i].x;
        reader->deltas[i][1] = (schar) icvCodeDeltas[i].y;
    }
}

/* Initialize sequence writer: */
void StartWriteSeq(int seq_flags, int header_size,
                 int elem_size, MemStorage* storage, SeqWriter* writer)
{
    if( !storage || !writer )
        fatal("NULL Pointer Error");

    Seq* seq = CreateSeq(seq_flags, header_size, elem_size, storage);
    StartAppendToSeq(seq, writer);
}


/* curvature: 0 - 1-curvature, 1 - k-cosine curvature. */
Seq* icvApproximateChainTC89(Chain* chain, int header_size, MemStorage* storage, int method)
{
    static const int abs_diff[] = { 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1 };

    PtInfo       temp;
    PtInfo       *array = malloc(sizeof(PtInfo)* (chain->total + 8)), *first = 0, *current = 0, *prev_current = 0;
    int             i, j, i1, i2, s, len;
    int             count = chain->total;

    ChainPtReader reader;
    SeqWriter     writer;
    Point         pt = chain->origin;

    assert(CV_IS_SEQ_CHAIN_CONTOUR(chain));
    assert(header_size >= (int)sizeof(Contour));

    StartWriteSeq((chain->flags & ~CV_SEQ_ELTYPE_MASK) | CV_SEQ_ELTYPE_POINT,
                     header_size, sizeof(Point), storage, &writer);

    if(chain->total == 0)
    {
        CV_WRITE_SEQ_ELEM( pt, writer );
        return EndWriteSeq( &writer );
    }

    reader.code = 0;
    StartReadChainPoints(chain, &reader);

    temp.next = 0;
    current = &temp;

    /* Pass 0.
       Restores all the digital curve points from the chain code.
       Removes the points (from the resultant polygon)
       that have zero 1-curvature */
    for(i = 0; i < count; i++)
    {
        int prev_code = *reader.prev_elem;

        reader.prev_elem = reader.ptr;
        CV_READ_CHAIN_POINT(pt, reader);

        /* calc 1-curvature */
        s = abs_diff[reader.code - prev_code + 7];

        if(method <= CV_CHAIN_APPROX_SIMPLE)
        {
            if(method == CV_CHAIN_APPROX_NONE || s != 0)
            {
                CV_WRITE_SEQ_ELEM(pt, writer);
            }
        }
        else
        {
            if( s != 0 )
                current = current->next = array + i;
            array[i].s = s;
            array[i].pt = pt;
        }
    }

    if(method <= CV_CHAIN_APPROX_SIMPLE)
        return EndWriteSeq(&writer);

    current->next = 0;

    len = i;
    current = temp.next;

    assert(current);

    /* Pass 1.
       Determines support region for all the remained points */
    do
    {
        Point pt0;
        int k, l = 0, d_num = 0;

        i = (int)(current - array);
        pt0 = array[i].pt;

        /* determine support region */
        for(k = 1;; k++)
        {
            int lk, dk_num;
            int dx, dy;
            Cv32suf d;

            assert(k <= len);

            /* calc indices */
            i1 = i - k;
            i1 += i1 < 0 ? len : 0;
            i2 = i + k;
            i2 -= i2 >= len ? len : 0;

            dx = array[i2].pt.x - array[i1].pt.x;
            dy = array[i2].pt.y - array[i1].pt.y;

            /* distance between p_(i - k) and p_(i + k) */
            lk = dx * dx + dy * dy;

            /* distance between p_i and the line (p_(i-k), p_(i+k)) */
            dk_num = (pt0.x - array[i1].pt.x) * dy - (pt0.y - array[i1].pt.y) * dx;
            d.f = (float)(((double) d_num) * lk - ((double) dk_num) * l);

            if( k > 1 && (l >= lk || ((d_num > 0 && d.i <= 0) || (d_num < 0 && d.i >= 0))))
                break;

            d_num = dk_num;
            l = lk;
        }

        current->k = --k;

        /* determine cosine curvature if it should be used */
        if( method == CV_CHAIN_APPROX_TC89_KCOS )
        {
            /* calc k-cosine curvature */
            for( j = k, s = 0; j > 0; j-- )
            {
                double temp_num;
                int dx1, dy1, dx2, dy2;
                Cv32suf sk;

                i1 = i - j;
                i1 += i1 < 0 ? len : 0;
                i2 = i + j;
                i2 -= i2 >= len ? len : 0;

                dx1 = array[i1].pt.x - pt0.x;
                dy1 = array[i1].pt.y - pt0.y;
                dx2 = array[i2].pt.x - pt0.x;
                dy2 = array[i2].pt.y - pt0.y;

                if( (dx1 | dy1) == 0 || (dx2 | dy2) == 0 )
                    break;

                temp_num = dx1 * dx2 + dy1 * dy2;
                temp_num =
                    (float) (temp_num /
                             sqrt(((double)dx1 * dx1 + (double)dy1 * dy1) *
                                   ((double)dx2 * dx2 + (double)dy2 * dy2)));
                sk.f = (float)(temp_num + 1.1);

                assert(0 <= sk.f && sk.f <= 2.2);
                if( j < k && sk.i <= s )
                    break;

                s = sk.i;
            }
            current->s = s;
        }
        current = current->next;
    }
    while(current != 0);

    prev_current = &temp;
    current = temp.next;

    /* Pass 2.
       Performs non-maxima suppression */
    do
    {
        int k2 = current->k >> 1;

        s = current->s;
        i = (int)(current - array);

        for(j = 1; j <= k2; j++)
        {
            i2 = i - j;
            i2 += i2 < 0 ? len : 0;

            if(array[i2].s > s)
                break;

            i2 = i + j;
            i2 -= i2 >= len ? len : 0;

            if(array[i2].s > s)
                break;
        }

        if(j <= k2)           /* exclude point */
        {
            prev_current->next = current->next;
            current->s = 0;     /* "clear" point */
        }
        else
            prev_current = current;
        current = current->next;
    }
    while(current != 0);

    /* Pass 3.
       Removes non-dominant points with 1-length support region */
    current = temp.next;
    assert(current);
    prev_current = &temp;

    do
    {
        if( current->k == 1 )
        {
            s = current->s;
            i = (int)(current - array);

            i1 = i - 1;
            i1 += i1 < 0 ? len : 0;

            i2 = i + 1;
            i2 -= i2 >= len ? len : 0;

            if( s <= array[i1].s || s <= array[i2].s )
            {
                prev_current->next = current->next;
                current->s = 0;
            }
            else
                prev_current = current;
        }
        else
            prev_current = current;
        current = current->next;
    }
    while( current != 0 );

    if( method == CV_CHAIN_APPROX_TC89_KCOS )
        goto copy_vect;

    /* Pass 4.
       Cleans remained couples of points */

    assert( temp.next );

    if( array[0].s != 0 && array[len - 1].s != 0 )      /* specific case */
    {
        for( i1 = 1; i1 < len && array[i1].s != 0; i1++ )
        {
            array[i1 - 1].s = 0;
        }
        if( i1 == len )
            goto copy_vect;     /* all points survived */
        i1--;

        for( i2 = len - 2; i2 > 0 && array[i2].s != 0; i2-- )
        {
            array[i2].next = 0;
            array[i2 + 1].s = 0;
        }
        i2++;

        if( i1 == 0 && i2 == len - 1 )  /* only two points */
        {
            i1 = (int)(array[0].next - array);
            array[len] = array[0];      /* move to the end */
            array[len].next = 0;
            array[len - 1].next = array + len;
        }
        temp.next = array + i1;
    }

    current = temp.next;
    first = prev_current = &temp;
    count = 1;

     /* do last pass */
    do
    {
        if( current->next == 0 || current->next - current != 1 )
        {
            if( count >= 2 )
            {
                if( count == 2 )
                {
                    int s1 = prev_current->s;
                    int s2 = current->s;

                    if( s1 > s2 || (s1 == s2 && prev_current->k <= current->k) )
                        /* remove second */
                        prev_current->next = current->next;
                    else
                        /* remove first */
                        first->next = current;
                }
                else
                    first->next->next = current;
            }
            first = current;
            count = 1;
        }
        else
            count++;
        prev_current = current;
        current = current->next;
    }
    while(current != 0);

    copy_vect:

    // gather points
    current = temp.next;
    assert(current);

    do
    {
        CV_WRITE_SEQ_ELEM( current->pt, writer );
        current = current->next;
    }
    while(current != 0);

    return EndWriteSeq(&writer);
}

Seq* FindNextContour(ContourScanner* scanner)
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

    if(mode == CV_RETR_FLOODFILL)
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

            ContourInfo *par_info = 0;
            ContourInfo *l_cinfo = 0;
            Seq *seq = 0;
            int is_hole = 0;
            Point origin;

             /* if not external contour */
            if( (!img_i && !(prev == 0 && prev == 1)) || (img_i && !(((prev & new_mask) != 0 || prev == 0) && (p & new_mask) == 0)))
            {
                /* check hole */
                if( (!img_i && (p != 0 || prev < 1)) ||
                        (img_i && ((prev & new_mask) != 0 || (p & new_mask) != 0)))
                    goto resume_scan;

                if(prev & new_mask)
                    lnbd.x = x-1;

                is_hole = 1;
            }

            if(mode == 0 && (is_hole || img0[lnbd.y * (size_t)(step) + lnbd.x] > 0))
                goto resume_scan;

            origin.y = y;
            origin.x = x - is_hole;

            /* find contour parent */
            if(mode <= 1 || (!is_hole && (mode == CV_RETR_CCOMP || mode == CV_RETR_FLOODFILL)) || lnbd.x <= 0)
                par_info = &(scanner->frame_info);

            else
            {
                int lval = (img0_i ?
                        img0_i[lnbd.y * (size_t)(step_i) + lnbd.x] :
                        (int)img0[lnbd.y * (size_t)(step) + lnbd.x]) & 0x7f;
                ContourInfo *cur = scanner->cinfo_table[lval];

                /* find the first bounding contour */
                while(cur)
                {
                    if((unsigned) (lnbd.x - cur->rect.x) < (unsigned) cur->rect.width &&
                            (unsigned) (lnbd.y - cur->rect.y) < (unsigned) cur->rect.height)
                    {
                        if(par_info)
                        {
                            if((img0_i &&
                                     icvTraceContour_32s(img0_i + par_info->origin.y * (size_t)(step_i) +
                                                          par_info->origin.x, step_i, img_i + lnbd.x,
                                                          par_info->is_hole) > 0) ||
                                    (!img0_i &&
                                     icvTraceContour(img0 + par_info->origin.y * (size_t)(step) +
                                                      par_info->origin.x, step, img + lnbd.x,
                                                      par_info->is_hole) > 0))
                                break;
                        }
                        par_info = cur;

                    }
                    cur = cur->next;
                }
                assert(par_info != 0)

                 /* if current contour is a hole and previous contour is a hole or
                       current contour is external and previous contour is external then
                       the parent of the contour is the parent of the previous contour else
                       the parent is the previous contour itself. */
                if(par_info->is_hole == is_hole)
                {
                    par_info = par_info->parent;
                    /* every contour must have a parent
                       (at least, the frame of the image) */
                    if(!par_info)
                        par_info = &(scanner->frame_info);
                }

                /* hole flag of the parent must differ from the flag of the contour */
                assert(par_info->is_hole != is_hole);
                if(par_info->contour == 0)        /* removed contour */
                    goto resume_scan;
            }

            lnbd.x = x - is_hole;

            SaveMemStoragePos(scanner->storage2, &(scanner->backup_pos));

            seq = CreateSeq(scanner->seq_type1, scanner->header_size1,
                                   scanner->elem_size1, scanner->storage1);
            seq->flags |= is_hole ? CV_SEQ_FLAG_HOLE : 0;

            /* initialize header */
            if(mode <= 1)
            {
                l_cinfo = &(scanner->cinfo_temp);
                icvFetchContour( img + x - is_hole, step,
                                 init_Point(origin.x + scanner->offset.x,
                                          origin.y + scanner->offset.y),
                                 seq, scanner->approx_method1);
            }
            else
            {
                union {ContourInfo* ci; SetElem* se;} v;
                v.ci = l_cinfo;
                SetAdd(scanner->cinfo_set, 0, &v.se);
                l_cinfo = v.ci;
                int lval;

                if(img_i)
                {
                    lval = img_i[x - is_hole] & 127;
                    icvFetchContourEx_32s(img_i + x - is_hole, step_i,
                                          init_Point(origin.x + scanner->offset.x,
                                                   origin.y + scanner->offset.y),
                                          seq, scanner->approx_method1,
                                          &(l_cinfo->rect));
                }
                else
                {
                    lval = nbd;
                    // change nbd
                    nbd = (nbd + 1) & 127;
                    nbd += nbd == 0 ? 3 : 0;
                    icvFetchContourEx(img + x-is_hole, step,
                                       init_Point(origin.x + scanner->offset.x,
                                                origin.y + scanner->offset.y),
                                       seq, scanner->approx_method1,
                                       lval, &(l_cinfo->rect));
                }
                l_cinfo->rect.x -= scanner->offset.x;
                l_cinfo->rect.y -= scanner->offset.y;

                l_cinfo->next = scanner->cinfo_table[lval];
                scanner->cinfo_table[lval] = l_cinfo;
            }
            l_cinfo->is_hole = is_hole;
            l_cinfo->contour = seq;
            l_cinfo->origin = origin;
            l_cinfo->parent = par_info;

            if(scanner->approx_method1 != scanner->approx_method2)
            {
                l_cinfo->contour = icvApproximateChainTC89((Chain *)seq,
                                                      scanner->header_size2,
                                                      scanner->storage2,
                                                      scanner->approx_method2);
                    ClearMemStorage(scanner->storage1);
            }

            l_cinfo->contour->v_prev = l_cinfo->parent->contour;

             if( par_info->contour == 0 )
            {
                l_cinfo->contour = 0;
                if(scanner->storage1 == scanner->storage2)
                {
                    RestoreMemStoragePos( scanner->storage1, &(scanner->backup_pos) );
                }
                else
                {
                    ClearMemStorage(scanner->storage1);
                }
                p = img[x];
                goto resume_scan;
            }

            SaveMemStoragePos( scanner->storage2, &(scanner->backup_pos2) );
            scanner->l_cinfo = l_cinfo;
            scanner->pt.x = !img_i ? x + 1 : x + 1 - is_hole;
            scanner->pt.y = y;
            scanner->lnbd = lnbd;
            scanner->img = (signed char*) img;
            scanner->nbd = nbd;
            return l_cinfo->contour;

        resume_scan:

            prev = p;
            /* update lnbd */
            if( prev & -2 )
            {
                lnbd.x = x;
            }
                            /* end of prev != p */
        }                   /* end of loop on x */

        lnbd.x = 0;
        lnbd.y = y + 1;
        x = 1;
        prev = 0;           /*end of loop on y */
    }                       

    return 0;
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

/* Release all blocks of the storage (or return them to parent, if any): */
static void icvDestroyMemStorage(MemStorage* storage)
{
    int k = 0;

    MemBlock* block;
    MemBlock* dst_top = 0;

    if(!storage)
        fatal("NULL Pointer Error");

    if(storage->parent)
        dst_top = storage->parent->top;

    for(block = storage->bottom; block != 0; k++)
    {
        MemBlock* temp = block;

        block = block->next;
        if(storage->parent)
        {
            if(dst_top)
            {
                temp->prev = dst_top;
                temp->next = dst_top->next;
                if(temp->next)
                    temp->next->prev = temp;
                dst_top = dst_top->next = temp;
            }
            else
            {
                dst_top = storage->parent->bottom = storage->parent->top = temp;
                temp->prev = temp->next = 0;
                storage->free_space = storage->block_size - sizeof(*temp);
            }
        }
        else
            free(&temp);
    }

    storage->top = storage->bottom = 0;
    storage->free_space = 0;
}

/* Release memory storage: */
void ReleaseMemStorage(MemStorage** storage)
{
    if(!storage)
        fatal("NULL Pointer Error");

    MemStorage* st = *storage;
    *storage = 0;
    if(st)
    {
        icvDestroyMemStorage( st );
        free(&st);
    }
}

Seq* cvEndFindContours(ContourScanner** _scanner)
{
    ContourScanner* scanner;
    Seq* first = 0;

    if(!_scanner)
        fatal("NULL Pointer Error");
    scanner = *_scanner;

    if(scanner)
    {
        icvEndProcessContour(scalar);

        if(scanner->storage1 != scanner->storage2)
            ReleaseMemStorage(&(scanner->storage1));

        if(scanner->cinfo_storage)
            ReleaseMemStorage(&(scanner->cinfo_storage));

        first = scanner->frame.v_next;
        free(_scanner);
    }
    return first;
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
        fatal("NULL double Seq pointer");

    *firstContour = 0;
    scanner = cvStartFindContours_Impl(img, storage, cntHeaderSize, mode, method, offset,
                                            needFillBorder);

    do
    {
        count++;
        contour = FindNextContour(scanner);
    }
    while(contour != 0);

    *firstContour = cvEndFindContours(&scanner);
    return count;
}

static inline void* AlignPtr(const void* ptr, int align/* 32 */)
{
    return (void*)(((size_t)ptr + align - 1) & ~(size_t)(align-1));
}

/* The function allocates space for at least one more sequence element.
   If there are free sequence blocks (seq->free_blocks != 0)
   they are reused, otherwise the space is allocated in the storage: */
static void icvGrowSeq(Seq *seq, int in_front_of)
{
    SeqBlock *block;

    if(!seq)
        fatal("NULL Pointer Error");

    block = seq->free_blocks;

    if(!block)
    {
        int elem_size = seq->elem_size;
        int delta_elems = seq->delta_elems;
        MemStorage *storage = seq->storage;

        if(seq->total >= delta_elems*4)
            SetSeqBlockSize(seq, delta_elems*2);

        if(!storage)
            fatal("The sequence has NULL storage pointer");

        /* If there is a free space just after last allocated block
           and it is big enough then enlarge the last block.
           This can happen only if the new block is added to the end of sequence: */
        if( (size_t)(ICV_FREE_PTR(storage) - seq->block_max) < CV_STRUCT_ALIGN &&
            storage->free_space >= seq->elem_size && !in_front_of )
        {
            int delta = storage->free_space / elem_size;

            delta = min(delta, delta_elems) * elem_size;
            seq->block_max += delta;
            storage->free_space = AlignLeft((int)(((signed char*)storage->top + storage->block_size) -
                                              seq->block_max), CV_STRUCT_ALIGN );
            return;
        }
        else
        {
            int delta = elem_size * delta_elems + ICV_ALIGNED_SEQ_BLOCK_SIZE;

            /* Try to allocate <delta_elements> elements: */
            if(storage->free_space < delta)
            {
                int small_block_size = max(1, delta_elems/3)*elem_size +
                                       ICV_ALIGNED_SEQ_BLOCK_SIZE;
                /* try to allocate smaller part */
                if(storage->free_space >= small_block_size + CV_STRUCT_ALIGN)
                {
                    delta = (storage->free_space - ICV_ALIGNED_SEQ_BLOCK_SIZE)/seq->elem_size;
                    delta = delta*seq->elem_size + ICV_ALIGNED_SEQ_BLOCK_SIZE;
                }
                else
                {
                    GoNextMemBlock(storage);
                    assert(storage->free_space >= delta);
                }
            }

            block = (SeqBlock*)MemStorageAlloc(storage, delta);
            block->data = (signed char*)AlignPtr(block + 1, CV_STRUCT_ALIGN);
            block->count = delta - ICV_ALIGNED_SEQ_BLOCK_SIZE;
            block->prev = block->next = 0;
        }
    }
    else
    {
        seq->free_blocks = block->next;
    }

    if(!(seq->first))
    {
        seq->first = block;
        block->prev = block->next = block;
    }
    else
    {
        block->prev = seq->first->prev;
        block->next = seq->first;
        block->prev->next = block->next->prev = block;
    }

    /* For free blocks the <count> field means
     * total number of bytes in the block.
     *
     * For used blocks it means current number
     * of sequence elements in the block:
     */
    assert(block->count % seq->elem_size == 0 && block->count > 0);

    if(!in_front_of)
    {
        seq->ptr = block->data;
        seq->block_max = block->data + block->count;
        block->start_index = block == block->prev ? 0 :
            block->prev->start_index + block->prev->count;
    }
    else
    {
        int delta = block->count / seq->elem_size;
        block->data += block->count;

        if(block != block->prev)
        {
            assert(seq->first->start_index == 0);
            seq->first = block;
        }
        else
        {
            seq->block_max = seq->ptr = block->data;
        }

        block->start_index = 0;

        for( ;; )
        {
            block->start_index += delta;
            block = block->next;
            if( block == seq->first )
                break;
        }
    }

    block->count = 0;
}

signed char* SeqPush(Seq *seq, const void *element)
{
    signed char *ptr = 0;
    size_t elem_size;

    if(!seq)
        fatal("NULL Pointer Error");

    elem_size = seq->elem_size;
    ptr = seq->ptr;

    if(ptr >= seq->block_max)
    {
        icvGrowSeq(seq, 0);

        ptr = seq->ptr;
        assert( ptr + elem_size <= seq->block_max /*&& ptr == seq->block_min */);
    }

    if(element)
        memcpy( ptr, element, elem_size );
    seq->first->prev->count++;
    seq->total++;
    seq->ptr = ptr + elem_size;

    return ptr;
}

void* NextTreeNode(TreeNodeIterator* treeIterator)
{
    TreeNode* prevNode = 0;
    TreeNode* node;
    int level;

    if(!treeIterator)
        fatal("NULL iterator pointer");

    prevNode = node = (TreeNode*)treeIterator->node;
    level = treeIterator->level;

    if(node)
    {
        if(node->v_next && level+1 < treeIterator->max_level)
        {
            node = node->v_next;
            level++;
        }
        else
        {
            while(node->h_next == 0)
            {
                node = node->v_prev;
                if( --level < 0 )
                {
                    node = 0;
                    break;
                }
            }
            node = node && treeIterator->max_level != 0 ? node->h_next : 0;
        }
    }

    treeIterator->node = node;
    treeIterator->level = level;
    return prevNode;
}

void cvInitTreeNodeIterator(TreeNodeIterator* treeIterator, const void* first, int max_level)
{
    if(!treeIterator || !first)
        fatal("NULL Pointer Error");

    if(max_level < 0)
        fatal("Out of Range Error");

    treeIterator->node = (void*)first;
    treeIterator->level = 0;
    treeIterator->max_level = max_level;
}

Seq* TreeToNodeSeq(const void* first, int header_size, MemStorage* storage)
{
    Seq* allseq = 0;
    TreeNodeIterator iterator;

    if( !storage )
        fatal("NULL storage pointer");

    allseq = CreateSeq(0, header_size, sizeof(first), storage);

    if(first)
    {
        cvInitTreeNodeIterator(&iterator, first, INT_MAX);

        for(;;)
        {
            void* node = NextTreeNode( &iterator );
            if(!node)
                break;
            SeqPush(allseq, &node);
        }
    }
    return allseq;
}

void findContours(Mat image0, vector** contours/* Point */, vector* hierarchy/* Scalar */, int mode, int method, Point offset)
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

    if(!_ccontours)
    {
        vector_free(*contours);
        return;
    }
    Seq* all_contours = TreeToNodeSeq(_ccontours, sizeof(Seq), storage);
    int i, total = (int)all_contours->total;
    createVectorOfVector(contours, total, 1, 0, -1, true, 0);
    
}

typedef struct Point
{
    int x;
    int y;
} Point ;

typedef struct Vec3i
{
    int val[3];
} Vec3i ;

Vec3i vec3i(int a, int b, int c)
{
    Vec3i v;
    v.val[0] = a;
    v.val[1] = b;
    v.val[2] = c;
    return v;
}

typedef struct floatPoint
{
    float x;
    float y;
} floatPoint ;

// area of a whole sequence
double contourArea(vector* _contour /* Point */, bool oriented)
{
    Mat contour = _contour.getMat();
    int npoints = checkVector(contour, 2, -1, true);
    int depth = depth(contour);
    assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    if(npoints == 0)
        return 0.;

    double a00 = 0;
    bool is_float = depth == CV_32F;
    const floatPoint* ptsi = (Point*)ptr(contour, 0);
    const floatPoint* ptsf = (floatPoint*)ptr(contour, 0);
    floatPoint prev = is_float ? ptsf[npoints-1] : init_floatPoint((float)ptsi[npoints-1].x, (float)ptsi[npoints-1].y);

    for(int i = 0; i < npoints; i++)
    {
        floatPoint p = is_float ? ptsf[i] : init_floatPoint((float)ptsi[i].x, (float)ptsi[i].y);
        a00 += (double)prev.x * p.y - (double)prev.y * p.x;
        prev = p;
    }

    a00 *= 0.5;
    if(!oriented)
        a00 = fabs(a00);

    return a00;
}

void approxPolyDP(Mat curve, vector* approxCurve /* Point */, double epsilon, bool closed)
{
    int npoints = checkVector(curve, 2, -1 ,true), depth = depth(curve);
    assert(npoints >= 0 && (depth == CV_32S || depth == CV_32F));

    if(npoints == 0)
    {
        vector_free(approxCurve);
        return;
    }

    Point* _stack = malloc(sizeof(Point)*npoints);
    Point* buf = malloc(sizeof(Point)*npoints);
    int nout = 0;

    if(depth == CV_32S)
        nout = approxPolyDP_int((Point*)ptr(curve, 0), npoints, buf, closed, epsilon, _stack);
    else if(depth == CV_32F)
        nout = approxPolyDP_float((floatPoint*)ptr(curve, 0), npoints, (floatPoint*)buf, closed, epsilon, _stack);
    else
        fatal("Unsupported Format");

    copyToVector(createMat(nout, 1, CV_MAKETYPE(depth, 2), buf, AUTO_STEP), approxCurve);
}

void copyToVector(Mat src, vector* dst)
{
    int dtype = 0;
    
    
    if(empty(src))
    {
        vector_free(dst);
        return;
    }

    size_t len = src.rows*src.cols > 0 ? src.rows + src.cols - 1 : 0;
    int mtype = CV_MAT_TYPE(type(src));

    vector* v = malloc(sizeof(vector));
    vector_init(v);
    for(int i = 0; i < vector_size(dst); i++)
        vector_add(v, (unsigned char*)vector_get(dst, i));

    vector_resize(v, len);
    Mat _dst = getMatfromVector(dst);
    
    if(src.data == _dst.data)
        return;

    if(src.rows > 0 && src.cols > 0)
    {
        _dst = reshape(_dst, 0, total(_dst));

        const unsigned char* sptr = src.data;
        unsigned char* dptr = _dst.data;

        Point sz = getContinuousSize(src, _dst);
        size_t len = sz.x * elemSize(src);

        for(; sz.y--; sptr += src.step, dptr += dst.step)
                memcpy(dptr, sptr, len);
    }
}

Mat reshape(Mat m, int new_cn, int new_rows)
{
    int cn = channels(m);

    if(new_cn == 0)
        new_cn = cn;

    int total_width = m.cols * cn;

    if((new_cn > total_width || total_width % new_cn != 0) && new_rows == 0)
        new_rows = m.rows * total_width/new_cn;

    if(new_rows != 0 && new_rows != rows)
    {
        int total_size = total_width * m.rows;
        if(!isContinuous(m))
            fatal("The matrix is not continuous, thus its number of rows can not be changed");

        if((unsigned)new_rows > (unsigned)total_size)
            fatal("Bad new number of rows");

        total_width = total_size / new_rows;

        if(total_width * new_rows != total_size)
            fatal("The total number of matrix elements is not divisible by the new number of rows");

        m.rows = new_rows;
        m.step[0] = total_width * elemSize1(m);
    }

    int new_width = total_width / new_cn;

    if(new_width * new_cn != total_width)
        fatal("The total width is not divisible by the new number of channels");

    m.cols = new_width;
    m.flags = (m.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
    m.step[1] = CV_ELEM_SIZE(m.flags);
    return m;
}

Mat getMatfromVector(vector* v)
{
    vector* u = malloc(sizeof(vector)), *w = malloc(sizeof(vector));
    vector_init(u);
    vector_init(w);

    for(int i = 0; i < vector_size(dst); i++)
    {
        vector_add(u, (unsigned char*)vector_get(v, i));
        vector_add(w, (int*)vector_get(v, i));
    }
    
    if(!vector_empty(u))
    {
        int szb = vector_size(u);
        int szi = vector_size(w);
        return createMat(1, szb, 0, vector_get(u, 0), AUTO_STEP);
    }
    else
    {
        Mat m;
        m.flags = MAGIC_VAL;
        m.rows = m.cols = 0;
        m.data = 0;
        m.datastart = 0;
        m.dataend = 0;
        m.datalimit = 0;
        m.step = 0;
        return m;
    }
}

static int approxPolyDP_int(const floatPoint* src_contour, int count0, floatPoint* dst_contour, bool is_closed0, double eps, AutoBuffer* _stack)
{
    #define PUSH_SLICE(slice) \
        if(top >= stacksz) \
        { \
            AutoBuffer_resize(_stack, stacksz*3/2); \
            stack = *_stack; \
            stacksz = _stack->sz; \
        } \
        stack[top++] = slice

    #define READ_PT(pt, pos) \
        pt = src_contour[pos]; \
        if(++pos >= count) pos = 0

    #define READ_DST_PT(pt, pos) \
        pt = dst_contour[pos]; \
        if(++pos >= count) pos = 0

    #define WRITE_PT(pt) \
        dst_contour[new_count++] = pt

    int             init_iters = 3;
    Point slice = init_Point(0, 0); /* Range */
    Point right_slice = init_Point(0, 0); /* Range */
    floatPoint start_pt = init_floatPoint((int)-1000000, (int)-1000000); 
    floatPoint end_pt = init_floatPoint(0, 0);
    floatPoint pt = init_floatPoint(0, 0);
    int i = 0, j, pos = 0, wpos, count = count0, new_count=0;
    int is_closed = is_closed0;
    bool le_eps = false;
    size_t top = 0, stacksz = _stack->sz;
    Point* stack = *_stack; /* Range */

    if(count == 0)
        return 0;

    eps *= eps;

    if(!is_closed)
    {
        right_slice.start = count;
        end_pt = src_contour[0];
        start_pt = src_contour[count-1];

        if(start_pt.x != end_pt.x || start_pt.y != end_pt.y)
        {
            slice.start = 0;
            slice.end = count - 1;
            PUSH_SLICE(slice);
        }
        else
        {
            is_closed = 1;
            init_iters = 1;
        }
    }

    if(is_closed)
    {
        // 1. Find approximately two farthest points of the contour
        right_slice.start = 0;

        for(i = 0; i < init_iters; i++)
        {
            double dist, max_dist = 0;
            pos = (pos + right_slice.start)%count;
            READ_PT(start_pt, pos);

            for(j = 1; j < count; j++)
            {
                double dx, dy;

                READ_PT(pt, pos);
                dx = pt.x - start_pt.x;
                dy = pt.y - start_pt.y;

                dist = dx * dx + dy * dy;

                if(dist > max_dist)
                {
                    max_dist = dist;
                    right_slice.start = j;
                }
            }

            le_eps = max_dist <= eps;
        }

        // 2. initialize the stack
        if(!le_eps)
        {
            right_slice.end = slice.start = pos % count;
            slice.end = right_slice.start = (right_slice.start + slice.start) % count;

            PUSH_SLICE(right_slice);
            PUSH_SLICE(slice);
        }
        else
            WRITE_PT(start_pt);
    }

    // 3. run recursive process
    while(top > 0)
    {
        slice = stack[--top];
        end_pt = src_contour[slice.end];
        pos = slice.start;
        READ_PT(start_pt, pos);

        if(pos != slice.end)
        {
            double dx, dy, dist, max_dist = 0;

            dx = end_pt.x - start_pt.x;
            dy = end_pt.y - start_pt.y;

            assert(dx != 0 || dy != 0);

            while(pos != slice.end)
            {
                READ_PT(pt, pos);
                dist = fabs((pt.y - start_pt.y) * dx - (pt.x - start_pt.x) * dy);

                if(dist > max_dist)
                {
                    max_dist = dist;
                    right_slice.start = (pos+count-1)%count;
                }
            }

            le_eps = max_dist * max_dist <= eps * (dx * dx + dy * dy);
        }
        else
        {
            le_eps = true;
            // read starting point
            start_pt = src_contour[slice.start];
        }

        if(le_eps)
        {
            WRITE_PT(start_pt);
        }
        else
        {
            right_slice.end = slice.end;
            slice.end = right_slice.start;
            PUSH_SLICE(right_slice);
            PUSH_SLICE(slice);
        }
    }

    if(!is_closed)
        WRITE_PT(src_contour[count-1]);

    // last stage: do final clean-up of the approximated contour -
    // remove extra points on the [almost] stright lines.
    is_closed = is_closed0;
    count = new_count;
    pos = is_closed ? count - 1 : 0;
    READ_DST_PT(start_pt, pos);
    wpos = pos;
    READ_DST_PT(pt, pos);

    for(i = !is_closed; i < count - !is_closed && new_count > 2; i++)
    {
        double dx, dy, dist, successive_inner_product;
        READ_DST_PT(end_pt, pos);

        dx = end_pt.x - start_pt.x;
        dy = end_pt.y - start_pt.y;
        dist = fabs((pt.x - start_pt.x)*dy - (pt.y - start_pt.y)*dx);
        successive_inner_product = (pt.x - start_pt.x) * (end_pt.x - pt.x) +
        (pt.y - start_pt.y) * (end_pt.y - pt.y);

        if(dist * dist <= 0.5*eps*(dx*dx + dy*dy) && dx != 0 && dy != 0 &&
           successive_inner_product >= 0)
        {
            new_count--;
            dst_contour[wpos] = start_pt = end_pt;
            if(++wpos >= count) wpos = 0;
            READ_DST_PT(pt, pos);
            i++;
            continue;
        }
        dst_contour[wpos] = start_pt = pt;
        if(++wpos >= count) wpos = 0;
        pt = end_pt;
    }

    if( !is_closed )
        dst_contour[wpos] = pt;

    return new_count;
}

static int approxPolyDP_float(const floatPoint* src_contour, int count0, floatPoint* dst_contour, bool is_closed0, double eps, AutoBuffer* _stack)
{
    #define PUSH_SLICE(slice) \
        if(top >= stacksz) \
        { \
            AutoBuffer_resize(_stack, stacksz*3/2); \
            stack = *_stack; \
            stacksz = _stack->sz; \
        } \
        stack[top++] = slice

    #define READ_PT(pt, pos) \
        pt = src_contour[pos]; \
        if( ++pos >= count ) pos = 0

    #define READ_DST_PT(pt, pos) \
        pt = dst_contour[pos]; \
        if( ++pos >= count ) pos = 0

    #define WRITE_PT(pt) \
        dst_contour[new_count++] = pt

    int             init_iters = 3;
    Point slice = init_Point(0, 0);
    Point right_slice = init_Point(0, 0);
    Point start_pt = init_Point((int)-1000000, (int)-1000000);
    Point end_pt = init_Point(0, 0);
    Point pt = init_Point(0, 0);
    int i = 0, j, pos = 0, wpos, count = count0, new_count=0;
    int is_closed = is_closed0;
    bool le_eps = false;
    size_t top = 0, stacksz = _stack->sz;
    Point* stack = *_stack;

    if(count == 0)
        return 0;

    eps *= eps;

    if(!is_closed)
    {
        right_slice.start = count;
        end_pt = src_contour[0];
        start_pt = src_contour[count-1];

        if(start_pt.x != end_pt.x || start_pt.y != end_pt.y)
        {
            slice.start = 0;
            slice.end = count - 1;
            PUSH_SLICE(slice);
        }
        else
        {
            is_closed = 1;
            init_iters = 1;
        }
    }

    if(is_closed)
    {
        // 1. Find approximately two farthest points of the contour
        right_slice.start = 0;

        for(i = 0; i < init_iters; i++)
        {
            double dist, max_dist = 0;
            pos = (pos + right_slice.start)%count;
            READ_PT(start_pt, pos);

            for(j = 1; j < count; j++)
            {
                double dx, dy;

                READ_PT(pt, pos);
                dx = pt.x - start_pt.x;
                dy = pt.y - start_pt.y;

                dist = dx * dx + dy * dy;

                if(dist > max_dist)
                {
                    max_dist = dist;
                    right_slice.start = j;
                }
            }

            le_eps = max_dist <= eps;
        }

        // 2. initialize the stack
        if(!le_eps)
        {
            right_slice.end = slice.start = pos % count;
            slice.end = right_slice.start = (right_slice.start + slice.start) % count;

            PUSH_SLICE(right_slice);
            PUSH_SLICE(slice);
        }
        else
            WRITE_PT(start_pt);
    }

    // 3. run recursive process
    while(top > 0)
    {
        slice = stack[--top];
        end_pt = src_contour[slice.end];
        pos = slice.start;
        READ_PT(start_pt, pos);

        if(pos != slice.end)
        {
            double dx, dy, dist, max_dist = 0;

            dx = end_pt.x - start_pt.x;
            dy = end_pt.y - start_pt.y;

            assert(dx != 0 || dy != 0);

            while(pos != slice.end)
            {
                READ_PT(pt, pos);
                dist = fabs((pt.y - start_pt.y) * dx - (pt.x - start_pt.x) * dy);

                if(dist > max_dist)
                {
                    max_dist = dist;
                    right_slice.start = (pos+count-1)%count;
                }
            }

            le_eps = max_dist * max_dist <= eps * (dx * dx + dy * dy);
        }
        else
        {
            le_eps = true;
            // read starting point
            start_pt = src_contour[slice.start];
        }

        if(le_eps)
        {
            WRITE_PT(start_pt);
        }
        else
        {
            right_slice.end = slice.end;
            slice.end = right_slice.start;
            PUSH_SLICE(right_slice);
            PUSH_SLICE(slice);
        }
    }

    if(!is_closed)
        WRITE_PT(src_contour[count-1]);

    // last stage: do final clean-up of the approximated contour -
    // remove extra points on the [almost] stright lines.
    is_closed = is_closed0;
    count = new_count;
    pos = is_closed ? count - 1 : 0;
    READ_DST_PT(start_pt, pos);
    wpos = pos;
    READ_DST_PT(pt, pos);

    for(i = !is_closed; i < count - !is_closed && new_count > 2; i++)
    {
        double dx, dy, dist, successive_inner_product;
        READ_DST_PT(end_pt, pos);

        dx = end_pt.x - start_pt.x;
        dy = end_pt.y - start_pt.y;
        dist = fabs((pt.x - start_pt.x)*dy - (pt.y - start_pt.y)*dx);
        successive_inner_product = (pt.x - start_pt.x) * (end_pt.x - pt.x) +
        (pt.y - start_pt.y) * (end_pt.y - pt.y);

        if(dist * dist <= 0.5*eps*(dx*dx + dy*dy) && dx != 0 && dy != 0 &&
           successive_inner_product >= 0)
        {
            new_count--;
            dst_contour[wpos] = start_pt = end_pt;
            if(++wpos >= count) wpos = 0;
            READ_DST_PT(pt, pos);
            i++;
            continue;
        }
        dst_contour[wpos] = start_pt = pt;
        if(++wpos >= count) wpos = 0;
        pt = end_pt;
    }

    if( !is_closed )
        dst_contour[wpos] = pt;

    return new_count;
}

void setThresholdDeta(ERFilterNM* filter, int thresholdDelta)
{
    filter->thresholdDelta = thresholdDelta;
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

