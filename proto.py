class Dataloader:
    def __init__(self, ulb_dataset, model, tgt_dist_matrix, batch_size=1, transform) -> None:
        '''
        ulb_dataset : torch dataset of unlabeled data
        model: torch model
        tgt_dist_matrix: KXK numpy array where each row is the class conditional sampling distribution
        '''
        self.ulb_dataset = ulb_dataset
        self.transform = transform
        self.model = model
        self.gen_pl()
        pass

    def gen_pl(self):
        '''
        this function mutates the ulb dataset with psuedolabels with the updated model object
        '''
        pass

    def get_dataloaders():
        '''
        generated K dataloader each having a sampling dist defined in tgt_dist_matrix
        '''
        pass
    def get_sample_weights():
        '''
        returns the individual sample weights for each u in ulb dataset given a class dependent target dist
        (row of tgt_dist_matrix)
        '''
        pass
    def get_batch(labels):
        '''
        returns a batch of ulb samples sampled from the corr. dist
        [dict[l].get_item() for l in labels] 
        '''
        # gets a list of labels (could be a list of torch cuda tensors, handle accordingly)
        # for every y in labels, the corr dataloader.next() returns u
        # we make a list of u, [self.transform(u) for u in U'] 
        # apply appropriate transforms
        # return batch u
