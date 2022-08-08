import os

def slow_rerank(args, qid, query, pids, passages):
    colbert = args.colbert
    inference = args.inference
    
    # query -> str
    # passages -> list(str)
    # len(passages) = 1000

    # Q_V = inference.query_tokenizer.encode([query])[0]
    # D_V = inference.doc_tokenizer.encode(passages)[0,:]


    # Q_V, Q = inference.queryFromText([query]*len(passages), bsize=args.bsize)
    # D_V, D_ = inference.docFromText(passages, bsize=args.bsize)

    # Q = inference.queryFromText([query]*len(passages), bsize=args.bsize)
    # D_ = inference.docFromText(passages, bsize=args.bsize)

    Q = inference.queryFromText([query]*len(passages))
    D_ = inference.docFromText(passages)

    # print(len(Q))
    # print(len(D_))

    # for idx, i in enumerate(D_):
    #     print(idx)
    #     print(i)

    # print("Q.shape")
    # print(Q.shape)
    # print("Q_V.shape")
    # print(Q_V.shape)
    # print("D_V.shape")
    # print(D_V.shape)
    # print("D_.shape")
    # print(D_.shape)

    # scores = colbert.score(Q_V, D_V, Q, D_).cpu()
    
    scores = colbert(qid, Q, D_).cpu()


    scores = scores.sort(descending=True)
    ranked = scores.indices.tolist()

    ranked_scores = scores.values.tolist()
    ranked_pids = [pids[position] for position in ranked]
    ranked_passages = [passages[position] for position in ranked]

    assert len(ranked_pids) == len(set(ranked_pids))

    return list(zip(ranked_scores, ranked_pids, ranked_passages))
