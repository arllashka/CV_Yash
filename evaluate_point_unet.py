def save_point_predictions(model, test_loader, device, save_dir, num_samples=10):
    """Save predictions with point visualization"""
    model.eval()
    os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)

    cat_predictions = []  # (IoU score, image, mask, pred, point, filename)
    dog_predictions = []  # (IoU score, image, mask, pred, point, filename)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating predictions"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            points = batch['point']
            point_heatmaps = batch['point_heatmap'].to(device)
            filenames = batch['filename']

            outputs = model(images, point_heatmaps)
            preds = torch.argmax(outputs, dim=1)

            # Calculate IoU for each image
            for idx, (image, mask, pred, point, filename) in enumerate(zip(images, masks, preds, points, filenames)):
                # For cats (class 1)
                if 1 in mask:
                    cat_mask = (mask == 1)
                    cat_pred = (pred == 1)
                    intersection = torch.logical_and(cat_mask, cat_pred).sum().float()
                    union = torch.logical_or(cat_mask, cat_pred).sum().float()
                    iou = (intersection / (union + 1e-8)).item()
                    cat_predictions.append((iou, image, mask, pred, point, filename))

                # For dogs (class 2)
                if 2 in mask:
                    dog_mask = (mask == 2)
                    dog_pred = (pred == 2)
                    intersection = torch.logical_and(dog_mask, dog_pred).sum().float()
                    union = torch.logical_or(dog_mask, dog_pred).sum().float()
                    iou = (intersection / (union + 1e-8)).item()
                    dog_predictions.append((iou, image, mask, pred, point, filename))

        # Sort by IoU and get top predictions
        cat_predictions.sort(key=lambda x: x[0], reverse=True)
        dog_predictions.sort(key=lambda x: x[0], reverse=True)

        cat_samples = cat_predictions[:num_samples]
        dog_samples = dog_predictions[:num_samples]

        def save_prediction(sample, prefix):
            iou, image, mask, pred, point, filename = sample

            plt.figure(figsize=(20, 5))

            # Original image with point
            plt.subplot(1, 4, 1)
            img_np = image.cpu().permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            # Debug print to see point structure
            print(f"Point structure: {point}")
            y, x = point[0], point[1]  # Modified this line
            plt.plot(x, y, 'rx', markersize=10)  # Add red X for point
            plt.title('Input Image with Point')
            plt.axis('off')

            # Point heatmap
            plt.subplot(1, 4, 2)
            heatmap = torch.zeros_like(mask, dtype=torch.float32)
            sigma = min(image.shape[1:]) / 16
            y_grid, x_grid = torch.meshgrid(torch.arange(image.shape[1]), torch.arange(image.shape[2]))
            heatmap = torch.exp(-((y_grid - y) ** 2 + (x_grid - x) ** 2) / (2 * sigma ** 2))
            plt.imshow(heatmap.cpu(), cmap='hot')
            plt.title('Point Heatmap')
            plt.axis('off')

            # Ground truth
            plt.subplot(1, 4, 3)
            plt.imshow(mask.cpu(), cmap='tab10', vmin=0, vmax=2)
            plt.title('Ground Truth')
            plt.axis('off')

            # Prediction
            plt.subplot(1, 4, 4)
            plt.imshow(pred.cpu(), cmap='tab10', vmin=0, vmax=2)
            plt.title(f'Prediction (IoU: {iou:.4f})')
            plt.axis('off')

            plt.savefig(os.path.join(save_dir, 'predictions', f'{prefix}_iou{iou:.4f}_{filename}.png'))
            plt.close()

        # Save predictions
        print("\nSaving predictions...")
        for idx, sample in enumerate(cat_samples):
            save_prediction(sample, f'cat_{idx + 1}')

        for idx, sample in enumerate(dog_samples):
            save_prediction(sample, f'dog_{idx + 1}')