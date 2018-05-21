import tensorflow as tf
from tensorflow.contrib import slim

def packup_tensors(v1,v2):


#unpack the first variable into a list of size 3 tensors
#there should be 5 tensors in the list
	change_shape = tf.unstack(v1)

#unpack the second variable into a list of size 3 tensors
#there should be 1 tensor in this list
	change_shape_2 = tf.unstack(v2)

#for each tensor in the second list, append it to the first list
	for i in range(len(change_shape_2)):
		change_shape.append(change_shape_2[i])

#repack the list of tensors into a single tensor
#the shape of this resultant tensor should be [6, 3]
	final = tf.stack(change_shape)
	return final



## This function is called in line 360 in RPN_model.py
def netgate(bev_proposal_rois, img_proposal_rois, is_training):

	## inputs are bev_rois and img_rois and model config, is_training, and 
	## it will return the netgate output

	
	#size of the roi proposals:
	#proposal_roi_crop_size = [rpn_config.rpn_proposal_roi_crop_size] * 2
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#    tf.print("bev_proposal_rois is:",bev_proposal_rois)
    tf.print(bev_proposal_rois)
    l2_weight_decay_ng = 0.0005
    keep_prob_ng = 0.5

    num_of_anchors=bev_proposal_rois.get_shape().as_list()[0]
    
	## not sure whether we need this
	## set up the regularizer
    if l2_weight_decay_ng > 0:
        weights_regularizer_ng = slim.l2_regularizer(l2_weight_decay_ng)
    else:
        weights_regularizer_ng = None
    fused_data=[]
    #flatten the bev and img rois to be 1*9
    bev_roi_flat=slim.flatten(bev_proposal_rois)
    img_roi_flat=slim.flatten(img_proposal_rois)

    ##fc layer for bev
    ##issue: what is the output size of the fully connected layer? input is 3*3 => 1*9
    layer_sizes_1 = 9
    layer_sizes_2 = 18
    layer_sizes_3 = 2


    ##issue: what is the def of scope?
    with slim.arg_scope([slim.fully_connected],
        weights_regularizer=weights_regularizer_ng):
    	
        bev_fc_layer = slim.fully_connected(bev_roi_flat, layer_sizes_1,
                                    scope='fc_bev')

        bev_fc_drop = slim.dropout(
                 bev_fc_layer,
                 keep_prob=keep_prob_ng,
                 is_training=is_training,
                 scope='fc_bev')
        ##fc layer for img
        img_fc_layer = slim.fully_connected(img_roi_flat, layer_sizes_1,
                                    scope='fc_img')

        img_fc_drop = slim.dropout(
                img_fc_layer,
                keep_prob=keep_prob_ng,
                is_training=is_training,
                scope='fc_img')

        ##concatenate bird view and image data
        concat_bev_img = tf.concat([bev_fc_drop, img_fc_drop], axis=1)

        ##TODO: another fc layer to output a reduced dimension
        concat_fc_layer = slim.fully_connected(concat_bev_img, layer_sizes_2,
                                            scope='fc_concat')

        concat_fc_drop = slim.dropout(
                concat_fc_layer,
                keep_prob=keep_prob_ng,
                is_training=is_training,
                scope='fc_concat')

        ##issue: what is Rectified Linear Unit in the Netgate paper? Ans: ReLu

        ##TODO: another fc layer to output a 2-D scalar vector s_b, s_i
        scalar_fc_layer = slim.fully_connected(concat_fc_drop, layer_sizes_3,
                                            scope='fc_scalar')

        scalar_fc_drop = slim.dropout(
                scalar_fc_layer,
                keep_prob=keep_prob_ng,
                is_training=is_training,
                scope='fc_scalar')
            ##TODO: fused data= s_b * bev_fc_drop + s_i * img_fc_drop

        scalar_0=scalar_fc_drop[:,0]
        scalar_1=scalar_fc_drop[:,1]

        scalar_0=tf.expand_dims(scalar_0,axis=-1)
        scalar_1=tf.expand_dims(scalar_1,axis=-1)

        fused_data=tf.multiply(scalar_0,bev_fc_drop) + tf.multiply(scalar_1,img_fc_drop)


##v_1 version, reshape to [80k,1,1,512]
#        fused_data=tf.expand_dims(fused_data,axis=1)
#        fused_data=tf.expand_dims(fused_data,axis=2)

##v_2 version, reshape to [80k,3,3,1]
        if isinstance(num_of_anchors,int):
            fused_data=tf.reshape(fused_data,[-1,3,3,1])
            print("fused is done")
        else:
            fused_data=bev_proposal_rois
            print("didn't fuse")

#        reshape_axis=fused_data.get_shape()
#        reshape_axis=list(map(int, reshape_axis))
#        fused_data=tf.reshape(fused_data,[?,1,1,512])

#        bev_result=tf.expand_dims(tf.multiply(scalar_fc_drop[0,0],bev_fc_drop[0]),0)
#        img_result=tf.expand_dims(tf.multiply(scalar_fc_drop[0,0],img_fc_drop[0]),0)
#        fused_result=tf.add(bev_result,img_result)

#        for i in range(1,num_of_anchors):
#            bev_res=tf.expand_dims(tf.multiply(scalar_fc_drop[i,0],bev_fc_drop[i]),0)
#            img_res=tf.expand_dims(tf.multiply(scalar_fc_drop[i,0],img_fc_drop[i]),0)
#            fused_res=tf.add(bev_res,img_res)
#            fused_result=packup_tensors(fused_result,fused_res)



    netgate_out= fused_data
    return netgate_out
	## first, figure out the size of bev/img_proposal_rois
	## 		this is defined in line95/96/97 in rpn_model.py
	## second, how to use with scope and slim to build up fc layers
	## confirm how to fuse the scalar*proposal_rois

