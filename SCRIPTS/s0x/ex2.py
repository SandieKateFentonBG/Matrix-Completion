# Instantiate model and optimizer

conv_net = True

if conv_net:
    model = ConvNet(3, num_classes).to(device)
    print("Convolutional model loaded")
    tr_accuracies_conv = np.zeros(num_epochs)
    val_accuracies_conv = np.zeros(num_epochs)
else:
    model = DenseNet(1024 * 3, num_classes).to(device)
    print("Dense model loaded")
    tr_accuracies_dense = np.zeros(num_epochs)
    val_accuracies_dense = np.zeros(num_epochs)
print("Number of trainable parameters: {}".format(count_parameters(model)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
best_val_acc = 0

start_time = time.time()

for epoch_nr in range(num_epochs):

    print("Epoch {}:".format(epoch_nr))

    # Train model
    running_loss = 0.0
    running_acc = 0.0
    for batch_data, batch_labels in trainloader:
        # Put data on device
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

        # Predict and get loss
        logits = model(batch_data)
        loss = criterion(logits, batch_labels)

        # Update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += compute_run_acc(logits, batch_labels)

    # Print results
    tr_acc = 100 * running_acc / len(trainloader.dataset)
    print('>> TRAIN: Epoch {} completed | tr_loss: {:.4f} | tr_acc: {:.2f}%'.format(
        epoch_nr, running_loss / len(trainloader.dataset), tr_acc))

    # Get validation results
    running_acc = 0
    with torch.no_grad():
        for batch_data, batch_labels in valloader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            logits = model(batch_data)
            running_acc += compute_run_acc(logits, batch_labels)

    val_acc = 100 * running_acc / len(valloader.dataset)
    print('>> VALIDATION: Epoch {} | val_acc: {:.2f}%'.format(epoch_nr, val_acc))

    if conv_net:
        tr_accuracies_conv[epoch_nr] = tr_acc
        val_accuracies_conv[epoch_nr] = val_acc
    else:
        tr_accuracies_dense[epoch_nr] = tr_acc
        val_accuracies_dense[epoch_nr] = val_acc

    # Save model if best accuracy on validation dataset until now
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), './cifar_net.pth')
        print('>> SAVE: Epoch {} | Model saved'.format(epoch_nr))

end_time = time.time()
print('Finished Training in {:.2f} seconds'.format(end_time - start_time))