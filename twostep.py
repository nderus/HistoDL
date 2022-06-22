# https://github.com/daib13/TwoStageVAE/blob/871862382746d619e7dc7527f3d02f924e1b0af5/network/two_stage_vae_model.py
#Notes
# - reconstruction loss is different? + crossentropy
# - I removed adapting learning rate step for now
class TwoStageVaeModel(object):
    def __init__(self, x, latent_dim=64, second_depth=3, second_dim=1024, cross_entropy_loss=False):
        self.raw_x = x
        self.x = tf.cast(self.raw_x, tf.float32) / 255.0 
        self.batch_size = x.get_shape().as_list()[0] # to add | ok
        self.latent_dim = latent_dim
        self.second_dim = second_dim  # to add |ok 
        self.second_depth = second_depth # to add | ok
        self.cross_entropy_loss = cross_entropy_loss 

        self.is_training = tf.placeholder(tf.bool, [], 'is_training')

        self.__build_network() # to add ?
        self.__build_loss()
        self.__build_summary()
        self.__build_optimizer() # to add

    def __build_network(self): # to add
        with tf.variable_scope('stage1'): # to add?
            self.build_encoder1()
            self.build_decoder1()
        with tf.variable_scope('stage2'): # to add?
            self.build_encoder2()
            self.build_decoder2()

    def __build_loss(self):
        HALF_LOG_TWO_PI = 0.91893 # to add? | for crossentropy, maybe later

        self.kl_loss1 = tf.reduce_sum(tf.square(self.mu_z) + tf.square(self.sd_z) - 2 * self.logsd_z - 1) / 2.0 / float(self.batch_size) # to add? instead of second reduce.
        if not self.cross_entropy_loss:
            self.gen_loss1 = tf.reduce_sum(tf.square((self.x - self.x_hat) / self.gamma_x) / 2.0 + self.loggamma_x + HALF_LOG_TWO_PI) / float(self.batch_size)
        else:
            self.gen_loss1 = -tf.reduce_sum(self.x * tf.log(tf.maximum(self.x_hat, 1e-8)) + (1-self.x) * tf.log(tf.maximum(1-self.x_hat, 1e-8))) / float(self.batch_size)
        self.loss1 = self.kl_loss1 + self.gen_loss1 

        self.kl_loss2 = tf.reduce_sum(tf.square(self.mu_u) + tf.square(self.sd_u) - 2 * self.logsd_u - 1) / 2.0 / float(self.batch_size)
        self.gen_loss2 = tf.reduce_sum(tf.square((self.z - self.z_hat) / self.gamma_z) / 2.0 + self.loggamma_z + HALF_LOG_TWO_PI) / float(self.batch_size)
        self.loss2 = self.kl_loss2 + self.gen_loss2 

    def __build_summary(self):
        with tf.name_scope('stage1_summary'):
            self.summary1 = []
            self.summary1.append(tf.summary.image('input', self.x))
            self.summary1.append(tf.summary.image('recon', self.x_hat))
            self.summary1.append(tf.summary.scalar('kl_loss', self.kl_loss1))
            self.summary1.append(tf.summary.scalar('gen_loss', self.gen_loss1))
            self.summary1.append(tf.summary.scalar('loss', self.loss1))
            self.summary1.append(tf.summary.scalar('gamma', self.gamma_x))
            self.summary1 = tf.summary.merge(self.summary1)

        with tf.name_scope('stage2_summary'):
            self.summary2 = []
            self.summary2.append(tf.summary.scalar('kl_loss', self.kl_loss2))
            self.summary2.append(tf.summary.scalar('gen_loss', self.gen_loss2))
            self.summary2.append(tf.summary.scalar('loss', self.loss2))
            self.summary2.append(tf.summary.scalar('gamma', self.gamma_z))
            self.summary2 = tf.summary.merge(self.summary2)

    def __build_optimizer(self): # to add
        all_variables = tf.global_variables()
        variables1 = [var for var in all_variables if 'stage1' in var.name]
        variables2 = [var for var in all_variables if 'stage2' in var.name]
        self.lr = tf.placeholder(tf.float32, [], 'lr')
        self.global_step = tf.get_variable('global_step', [], tf.int32, tf.zeros_initializer(), trainable=False)
        self.opt1 = tf.train.AdamOptimizer(self.lr).minimize(self.loss1, self.global_step, var_list=variables1)
        self.opt2 = tf.train.AdamOptimizer(self.lr).minimize(self.loss2, self.global_step, var_list=variables2)
        
    def build_encoder2(self): # to add
        with tf.variable_scope('encoder'):
            t = self.z 
            for i in range(self.second_depth):
                t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
            t = tf.concat([self.z, t], -1)
        
            self.mu_u = tf.layers.dense(t, self.latent_dim, name='mu_u')
            self.logsd_u = tf.layers.dense(t, self.latent_dim, name='logsd_u')
            self.sd_u = tf.exp(self.logsd_u)
            self.u = self.mu_u + self.sd_u * tf.random_normal([self.batch_size, self.latent_dim])
        
    def build_decoder2(self): # to add
        with tf.variable_scope('decoder'):
            t = self.u 
            for i in range(self.second_depth):
                t = tf.layers.dense(t, self.second_dim, tf.nn.relu, name='fc'+str(i))
            t = tf.concat([self.u, t], -1)

            self.z_hat = tf.layers.dense(t, self.latent_dim, name='z_hat')
            self.loggamma_z = tf.get_variable('loggamma_z', [], tf.float32, tf.zeros_initializer())
            self.gamma_z = tf.exp(self.loggamma_z)

    def extract_posterior(self, sess, x):
        num_sample = np.shape(x)[0]
        num_iter = math.ceil(float(num_sample) / float(self.batch_size))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        mu_z, sd_z = [], []
        for i in range(num_iter):
            mu_z_batch, sd_z_batch = sess.run([self.mu_z, self.sd_z], feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
            mu_z.append(mu_z_batch)
            sd_z.append(sd_z_batch)
        mu_z = np.concatenate(mu_z, 0)[0:num_sample]
        sd_z = np.concatenate(sd_z, 0)[0:num_sample]
        return mu_z, sd_z

    def step(self, stage, input_batch, lr, sess, writer=None, write_iteration=600): # to add?
        if stage == 1:
            loss, summary, _ = sess.run([self.loss1, self.summary1, self.opt1], feed_dict={self.raw_x: input_batch, self.lr: lr, self.is_training: True})
        elif stage == 2:
            loss, summary, _ = sess.run([self.loss2, self.summary2, self.opt2], feed_dict={self.z: input_batch, self.lr: lr, self.is_training: True})
        else:
            raise Exception('Wrong stage {}.'.format(stage))
        global_step = self.global_step.eval(sess)
        if global_step % write_iteration == 0 and writer is not None:
            writer.add_summary(summary, global_step)
        return loss 

    def reconstruct(self, sess, x):
        num_sample = np.shape(x)[0]
        num_iter = math.ceil(float(num_sample) / float(self.batch_size))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        recon_x = []
        for i in range(num_iter):
            recon_x_batch = sess.run(self.x_hat, feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
            recon_x.append(recon_x_batch)
        recon_x = np.concatenate(recon_x, 0)[0:num_sample]
        return recon_x 

    def generate(self, sess, num_sample, stage=2):
        num_iter = math.ceil(float(num_sample) / float(self.batch_size))
        gen_samples = []
        for i in range(num_iter):
            if stage == 2: # to add?
                # u ~ N(0, I)
                u = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
                # z ~ N(f_2(u), \gamma_z I)
                z, gamma_z = sess.run([self.z_hat, self.gamma_z], feed_dict={self.u: u, self.is_training: False})
                z = z + gamma_z * np.random.normal(0, 1, [self.batch_size, self.latent_dim])
            else:
                z = np.random.normal(0, 1, [self.batch_size, self.latent_dim])
            # x = f_1(z)
            x = sess.run(self.x_hat, feed_dict={self.z: z, self.is_training: False})
            gen_samples.append(x)
        gen_samples = np.concatenate(gen_samples, 0)
        return gen_samples[0:num_sample]


###

class TwoStageCVAE(keras.Model):
    def __init__(self, encoder1, decoder1, encoder2, decoder2, beta, shape, econd_depth=3, second_dim=1024, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder1 = encoder1
        self.decoder1 = decoder1
        self.encoder2 = self.build_encoder2
        self.decoder2 = self.build_decoder2
        self.beta = beta
        self.shape = shape
        self.second_dim = second_dim  # to add
        self.second_depth = second_depth # to add
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        #
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_loss")
        self.val_reconstruction_loss_tracker = keras.metrics.Mean(
            name="val_reconstruction_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.val_total_loss_tracker,
            self.val_reconstruction_loss_tracker,
            self.val_kl_loss_tracker,

        ]
       
    def call(self, inputs):
        _, input_label, conditional_input = self.conditional_input(inputs)
        z_mean, z_log_var = self.encoder(conditional_input)
        self.z_cond = self.sampling(z_mean, z_log_var, input_label)
        return self.decoder(self.z_cond)

    def build_encoder2(self): # to add
        with tf.variable_scope('encoder'):
            t = self.z_cond # z is output of reparametrization trick
            for i in range(self.second_depth):
                t = layers.Dense(t, self.second_dim, 'relu', name='fc'+str(i))
            t = tf.concat([self.z_cond, t], -1)
        
            self.mu_u = layers.Dense(t, self.latent_dim, name='mu_u')
            self.logsd_u = layers.Dense(t, self.latent_dim, name='logsd_u')
            self.sd_u = tf.exp(self.logsd_u)
            self.u = self.mu_u + self.sd_u * tf.random_normal([self.batch_size, self.latent_dim]) #reparametrization again
        
    def build_decoder2(self): # to add
        with tf.variable_scope('decoder'):
            t = self.u 
            for i in range(self.second_depth):
                t = layers.Dense(t, self.second_dim, 'relu', name='fc'+str(i))
            t = tf.concat([self.u, t], -1)

            self.z_hat = layers.Dense(t, self.latent_dim, name='z_hat')
            self.loggamma_z = tf.get_variable('loggamma_z', [], tf.float32, tf.zeros_initializer())
            self.gamma_z = tf.exp(self.loggamma_z)
    
    def extract_posterior(self, sess, x):
        num_sample = np.shape(x)[0]
        num_iter = math.ceil(float(num_sample) / float(self.batch_size))
        x_extend = np.concatenate([x, x[0:self.batch_size]], 0)
        mu_z, sd_z = [], []
        for i in range(num_iter):
            mu_z_batch, sd_z_batch = sess.run([self.mu_z, self.sd_z], feed_dict={self.raw_x: x_extend[i*self.batch_size:(i+1)*self.batch_size], self.is_training: False})
            mu_z.append(mu_z_batch)
            sd_z.append(sd_z_batch)
        mu_z = np.concatenate(mu_z, 0)[0:num_sample]
        sd_z = np.concatenate(sd_z, 0)[0:num_sample]
        return mu_z, sd_z
    
    def conditional_input(self, inputs, label_size=10): 
  
        image_size = [self.shape[0], self.shape[1], self.shape[2]]
    
        input_img = layers.InputLayer(input_shape=image_size,
                                      dtype ='float32')(inputs[0])
        input_label = layers.InputLayer(input_shape=(label_size, ),
                                        dtype ='float32')(inputs[1])

        labels = tf.reshape(inputs[1], [-1, 1, 1, label_size])
        labels = tf.cast(labels, dtype='float32')
        ones = tf.ones([inputs[0].shape[0]] + image_size[0:-1] + [label_size]) 
        labels = ones * labels
        conditional_input = layers.Concatenate(axis=3)([input_img, labels]) 
        return  input_img, input_label, conditional_input

    def sampling(self, z_mean, z_log_var, input_label):
        if len(input_label.shape) == 1:
            input_label = np.expand_dims(input_label, axis=0)

        eps = tf.random.normal(tf.shape(z_log_var), dtype=tf.float32,
                               mean=0., stddev=1.0, name='epsilon')
        z = z_mean + tf.exp(z_log_var / 2) * eps
        z_cond = tf.concat([z, input_label], axis=1)
        return z_cond

    def train_step(self, data):

        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
        
            input_img, input_label, conditional_input = self.conditional_input(data)
            z_mean, z_log_var = self.encoder(conditional_input)
            self.latent_var.append(z_log_var)
            z_cond = self.sampling(z_mean, z_log_var, input_label)
            reconstruction = self.decoder(z_cond)
            #reconstruction_loss = np.prod(self.shape) * tf.keras.losses.MSE(tf.keras.backend.flatten(input_img),
            #                        tf.keras.backend.flatten(reconstruction))
            HALF_LOG_TWO_PI = 0.91893 # to add?
            reconstruction_loss = tf.reduce_sum(
                 keras.losses.MSE(input_img, # removed np.prod(self.shape) *
                                    reconstruction), axis=(1, 2))            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)
                      - tf.exp(z_log_var))
            #kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) # was just reduce_sum
            kl_loss = tf.reduce_sum(kl_loss, axis=1) #sum over encoded dimensiosn, average over batch
            kl_loss = self.beta * kl_loss
            total_loss = reconstruction_loss + kl_loss
            total_loss = tf.reduce_mean(total_loss) #not necessary since I added red mean in kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        #wandb.log({"loss": total_loss, "reconstructon_loss": reconstruction_loss, "kl_loss": kl_loss,})
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        input_img, input_label, conditional_input = self.conditional_input(data)
        z_mean, z_log_var = self.encoder(conditional_input)
        z_cond = self.sampling(z_mean, z_log_var, input_label)
        reconstruction = self.decoder(z_cond)
        reconstruction_loss = tf.reduce_sum(
                 keras.losses.MSE(input_img,
                                    reconstruction), axis=(1, 2))   
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean)
                  - tf.exp(z_log_var))
        kl_loss = tf.reduce_sum(kl_loss, axis=1)
        kl_loss = self.beta * kl_loss
        total_loss =  reconstruction_loss + kl_loss
        total_loss = tf.reduce_mean(total_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return{
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
        }


num_sample = np.shape(train_x)[0] 
iteration_per_epoch = num_sample // batch_size 

# first stage
for epoch in range(args.epochs):
    np.random.shuffle(x)
    #lr = args.lr if args.lr_epochs <= 0 else args.lr * math.pow(args.lr_fac, math.floor(float(epoch) / float(args.lr_epochs)))
    lr = learning_rate
    epoch_loss = 0
    for j in range(iteration_per_epoch):
        image_batch = x[j *args.batch_size:(j+1)*args.batch_size]
        loss = model.step(1, image_batch, lr, sess, writer, args.write_iteration)
        epoch_loss += loss 
    epoch_loss /= iteration_per_epoch

    print('Date: {date}\t'
            'Epoch: [Stage 1][{0}/{1}]\t'
            'Loss: {2:.4f}.'.format(epoch, args.epochs, epoch_loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))
saver.save(sess, os.path.join(model_path, 'stage1'))

 # second stage
        mu_z, sd_z = model.extract_posterior(sess, x) #change this so it returns batches?
        idx = np.arange(num_sample)
        for epoch in range(args.epochs2):
            np.random.shuffle(idx)
            mu_z = mu_z[idx]
            sd_z = sd_z[idx]
            lr = args.lr2 if args.lr_epochs2 <= 0 else args.lr2 * math.pow(args.lr_fac2, math.floor(float(epoch) / float(args.lr_epochs2)))
            epoch_loss = 0
            for j in range(iteration_per_epoch):
                mu_z_batch = mu_z[j*args.batch_size:(j+1)*args.batch_size]
                sd_z_batch = sd_z[j*args.batch_size:(j+1)*args.batch_size]
                z_batch = mu_z_batch + sd_z_batch * np.random.normal(0, 1, [args.batch_size, args.latent_dim])
                loss = model.step(2, z_batch, lr, sess, writer, args.write_iteration) #add step 2
                epoch_loss += loss 
            epoch_loss /= iteration_per_epoch

            print('Date: {date}\t'
                  'Epoch: [Stage 2][{0}/{1}]\t'
                  'Loss: {2:.4f}.'.format(epoch, args.epochs2, epoch_loss, date=time.strftime('%Y-%m-%d %H:%M:%S')))
        saver.save(sess, os.path.join(model_path, 'stage2'))
    else:
        saver.restore(sess, os.path.join(model_path, 'stage2'))